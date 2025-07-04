import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import wandb
import os
from data.dataset import NeuronVisionDataset
from evaluation.wandb_helpers import log_diagnostics_to_wandb
from sklearn.metrics import log_loss

def train_single_fold(
    fold_idx,
    train_stim_idxs,
    test_stim_idxs,
    model_class,
    model_config,
    dataset_path_dict,
    wandb_config,
    num_epochs,
    batch_size,
    learning_rate,
    device
):
    # Load full dataset
    full_dataset = NeuronVisionDataset(
        embeddings_path=dataset_path_dict["embeddings"],
        neural_data_path=dataset_path_dict["neural"]
    )

    trials_per_stimulus = 50

    def stim_to_trial_indices(stim_idxs):
        return np.concatenate([
            np.arange(stim * trials_per_stimulus, (stim + 1) * trials_per_stimulus)
            for stim in stim_idxs
        ])

    train_trial_idxs = stim_to_trial_indices(train_stim_idxs)
    test_trial_idxs = stim_to_trial_indices(test_stim_idxs)

    def trial_to_dataset_indices(trial_idxs, num_neurons):
        return [
            trial * num_neurons + neuron
            for trial in trial_idxs
            for neuron in range(num_neurons)
        ]

    train_idx_expanded = trial_to_dataset_indices(train_trial_idxs, full_dataset.num_neurons)
    test_idx_expanded = trial_to_dataset_indices(test_trial_idxs, full_dataset.num_neurons)

    train_loader = DataLoader(Subset(full_dataset, train_idx_expanded), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(full_dataset, test_idx_expanded), batch_size=batch_size)

    wandb_run = wandb.init(project=wandb_config["project"], config=wandb_config, name=f"{wandb_config['name']}_fold{fold_idx}")

    model = model_class(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        all_preds, all_labels = [], []
        for batch in train_loader:
            optimizer.zero_grad()

            img = batch["image_embedding"].to(device)
            neuron_idx = batch["neuron_idx"].to(device)
            target = batch["response"].to(device).float()

            output = model(img, neuron_idx).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            all_preds.append(output.detach().cpu().numpy())
            all_labels.append(target.cpu().numpy())

        pred_probs_train = np.concatenate(all_preds)
        true_labels_train = np.concatenate(all_labels)
        avg_ll = -log_loss(true_labels_train, pred_probs_train, labels=[0, 1])
        wandb.log({f"fold{fold_idx}/train/avg_log_likelihood": avg_ll, "epoch": epoch})

    # Evaluation
    model.eval()
    for phase, loader in [("train", train_loader), ("test", test_loader)]:
        all_preds, all_labels, all_neurons = [], [], []

        with torch.no_grad():
            for batch in loader:
                img = batch["image_embedding"].to(device)
                neuron_idx = batch["neuron_idx"].to(device)
                target = batch["response"].to(device).float()

                output = model(img, neuron_idx).squeeze()
                all_preds.append(output.cpu().numpy())
                all_labels.append(target.cpu().numpy())
                all_neurons.append(neuron_idx.cpu().numpy())

        pred_probs = np.concatenate(all_preds)
        true_labels = np.concatenate(all_labels)
        neuron_ids = np.concatenate(all_neurons)

        log_diagnostics_to_wandb(pred_probs, true_labels, neuron_ids, phase=f"fold{fold_idx}/{phase}")

        if phase == "test":
            os.makedirs("wandb_folds", exist_ok=True)
            np.savez(
                f"wandb_folds/fold{fold_idx}_test_preds.npz",
                probs=pred_probs,
                labels=true_labels,
                neuron_ids=neuron_ids
            )

    wandb.finish()

def run_training(
    model_class,
    model_config,
    dataset_path_dict,
    wandb_config,
    num_epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    num_folds=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    from multiprocessing import Process

    num_stimuli = 118
    trials_per_stimulus = 50

    stimulus_indices = np.arange(num_stimuli)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    processes = []

    for fold_idx, (train_stim_idxs, test_stim_idxs) in enumerate(kf.split(stimulus_indices)):
        if fold_idx >= 5:
            break  # Limit to 5 folds in parallel

        p = Process(
            target=train_single_fold,
            args=(
                fold_idx,
                train_stim_idxs,
                test_stim_idxs,
                model_class,
                model_config,
                dataset_path_dict,
                wandb_config,
                num_epochs,
                batch_size,
                learning_rate,
                device
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import wandb

from data.dataset import NeuronVisionDataset
from evaluation.wandb_helpers import log_diagnostics_to_wandb

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
    # Initialize wandb
    wandb.init(project=wandb_config["project"], config=wandb_config)

    # Load full dataset once
    full_dataset = NeuronVisionDataset(
        embeddings_path=dataset_path_dict["embeddings"],
        neural_data_path=dataset_path_dict["neural"]
    )

    # Constants
    num_stimuli = 118
    trials_per_stimulus = 50

    stimulus_indices = np.arange(num_stimuli)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold_idx, (train_stim_idxs, test_stim_idxs) in enumerate(kf.split(stimulus_indices)):
        wandb.run.name = f"{wandb_config['name']}_fold{fold_idx}"
        print(f"\n🔁 Fold {fold_idx + 1}/{num_folds}")

        # Map stimulus indices to trial indices
        def stim_to_trial_indices(stim_idxs):
            return np.concatenate([
                np.arange(stim * trials_per_stimulus, (stim + 1) * trials_per_stimulus)
                for stim in stim_idxs
            ])

        train_trial_idxs = stim_to_trial_indices(train_stim_idxs)
        test_trial_idxs = stim_to_trial_indices(test_stim_idxs)

        # Expand trial indices to dataset indices (trial x neuron)
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

        # Instantiate a new model for each fold
        model = model_class(**model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCELoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()

                img = batch["image_embedding"].to(device)
                neuron_idx = batch["neuron_idx"].to(device)
                target = batch["response"].to(device).float()

                output = model(img, neuron_idx).squeeze()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(target)

            avg_train_loss = total_loss / len(train_loader.dataset)
            wandb.log({f"fold{fold_idx}/train/loss": avg_train_loss, "epoch": epoch})

        # === EVALUATION ===
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

    wandb.finish()

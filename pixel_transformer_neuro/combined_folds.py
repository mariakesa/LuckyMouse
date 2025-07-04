import numpy as np
import os
from evaluation.wandb_helpers import log_diagnostics_to_wandb


def load_fold_predictions(folder_path, fold_count=5):
    all_probs = []
    all_labels = []
    all_neurons = []

    for fold_idx in range(fold_count):
        file_path = os.path.join(folder_path, f"fold{fold_idx}_test_preds.npz")
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue

        data = np.load(file_path)
        all_probs.append(data['probs'])
        all_labels.append(data['labels'])
        all_neurons.append(data['neuron_ids'])

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    neuron_ids = np.concatenate(all_neurons)

    return probs, labels, neuron_ids


def main():
    folder = "wandb_folds"
    probs, labels, neuron_ids = load_fold_predictions(folder, fold_count=5)

    import wandb
    wandb.init(project="pixel-transformer-neuro", name="test_combined_summary")
    log_diagnostics_to_wandb(probs, labels, neuron_ids, phase="test_combined")
    wandb.finish()


if __name__ == "__main__":
    main()

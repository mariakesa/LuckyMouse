import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

def log_diagnostics_to_wandb(pred_probs, true_labels, neuron_ids=None, phase="train"):
    """
    Logs diagnostic metrics and plots to Wandb.

    Args:
        pred_probs (np.ndarray): shape (N,), predicted probabilities
        true_labels (np.ndarray): shape (N,), binary ground-truth labels
        neuron_ids (np.ndarray or list): shape (N,), neuron ID for each prediction (optional)
        phase (str): "train" or "test"
    """

    pred_probs = np.asarray(pred_probs)
    true_labels = np.asarray(true_labels)

    # Average Log-likelihood (comparable across train/test)
    avg_log_likelihood = -log_loss(true_labels, pred_probs, labels=[0, 1])
    wandb.log({f"{phase}/avg_log_likelihood": avg_log_likelihood})

    # Event-only average log-likelihood
    spike_mask = true_labels == 1
    if spike_mask.sum() > 0:
        avg_event_log_likelihood = -log_loss(true_labels[spike_mask], pred_probs[spike_mask], labels=[0, 1])
        wandb.log({f"{phase}/avg_event_log_likelihood": avg_event_log_likelihood})

    # Confusion Matrix
    preds_binary = (pred_probs >= 0.5).astype(int)
    cm = confusion_matrix(true_labels, preds_binary)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{phase.capitalize()} Confusion Matrix")
    wandb.log({f"{phase}/confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    # Histogram of predicted probabilities
    wandb.log({
        f"{phase}/prob_histogram_all": wandb.Histogram(pred_probs),
    })
    if spike_mask.sum() > 0:
        wandb.log({
            f"{phase}/prob_histogram_spikes": wandb.Histogram(pred_probs[spike_mask])
        })

    # ROC AUC and PR AUC
    try:
        roc_auc = roc_auc_score(true_labels, pred_probs)
        ap_score = average_precision_score(true_labels, pred_probs)
        wandb.log({
            f"{phase}/roc_auc": roc_auc,
            f"{phase}/average_precision": ap_score,
        })
    except:
        pass  # Handle edge cases with no positive labels etc.

    # Per-neuron event log-likelihood (optional)
    if neuron_ids is not None:
        neuron_ids = np.asarray(neuron_ids)
        unique_neurons = np.unique(neuron_ids)
        per_neuron_ll = []

        for n in unique_neurons:
            mask = neuron_ids == n
            if np.any(mask & spike_mask):
                try:
                    ll = -log_loss(true_labels[mask & spike_mask], pred_probs[mask & spike_mask], labels=[0, 1])
                    per_neuron_ll.append(ll)
                except:
                    per_neuron_ll.append(np.nan)
            else:
                per_neuron_ll.append(np.nan)

        mean_ll = np.nanmean(per_neuron_ll)
        wandb.log({f"{phase}/avg_event_log_likelihood_per_neuron": mean_ll})

        # Optional: heatmap
        fig, ax = plt.subplots(figsize=(12, 1))
        sns.heatmap(np.expand_dims(per_neuron_ll, axis=0), cmap="magma", ax=ax, cbar=True)
        ax.set_yticks([])
        ax.set_title(f"{phase.capitalize()} Per-Neuron Event Log-Likelihood")
        wandb.log({f"{phase}/per_neuron_event_ll_heatmap": wandb.Image(fig)})
        plt.close(fig)

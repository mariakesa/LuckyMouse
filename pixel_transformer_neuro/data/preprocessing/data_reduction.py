import numpy as np
import os

def reduce_trials_to_probabilities(neural_data_path, output_path, trials_per_image=50):
    """
    Reduces (neurons, 5900) binary spike data to (neurons, 118) firing probabilities.
    """

    data = np.load(neural_data_path)  # shape: (N_neurons, 5900)
    num_neurons, total_trials = data.shape
    num_images = total_trials // trials_per_image

    assert total_trials % trials_per_image == 0, "Trials don't divide evenly into images."

    # Reshape: (neurons, 118, 50)
    reshaped = data.reshape(num_neurons, num_images, trials_per_image)

    # Compute mean over trials axis → (neurons, 118)
    firing_probs = reshaped.mean(axis=2)

    print(f"Reduced from shape {data.shape} → {firing_probs.shape}")
    np.save(output_path, firing_probs)
    print(f"Saved averaged probabilities to {output_path}")

if __name__=='__main__':
    reduce_trials_to_probabilities("pixel_transformer_neuro/data/processed/hybrid_neural_responses.npy", "pixel_transformer_neuro/data/processed/hybrid_neural_responses_reduced.npy", trials_per_image=50)
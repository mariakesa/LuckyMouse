import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class NeuronVisionDataset(Dataset):
    def __init__(self, embeddings_path, neural_data_path):
        with open(embeddings_path, "rb") as f:
            embeddings_raw = pickle.load(f)

        self.image_embeddings = torch.tensor(
            embeddings_raw['natural_scenes'], dtype=torch.float32
        )  # (118, D)

        # Use memory-mapped neural data
        self.neural_events = np.load(neural_data_path, mmap_mode='r')  # (N_neurons, 5900)

        self.num_stimuli = 118
        self.trials_per_stimulus = 1
        self.num_trials = self.num_stimuli * self.trials_per_stimulus
        self.num_neurons = self.neural_events.shape[0]

        # Lazy trial index → stimulus index mapping
        self.trial_to_stimulus = torch.arange(self.num_stimuli).repeat_interleave(self.trials_per_stimulus)

        # We don’t precompute all pairs — we just index dynamically
        self.total_samples = self.num_trials * self.num_neurons

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        trial_idx = idx // self.num_neurons
        neuron_idx = idx % self.num_neurons

        stimulus_idx = self.trial_to_stimulus[trial_idx]
        image_embedding = self.image_embeddings[stimulus_idx]
        response = float(self.neural_events[neuron_idx, trial_idx] > 0)

        return {
            "image_embedding": image_embedding,
            "neuron_idx": neuron_idx,
            "stimulus_idx": stimulus_idx,
            "response": response
        }


if __name__=="__main__":
    from torch.utils.data import DataLoader

    dataset = NeuronVisionDataset(
        embeddings_path="data/processed/google_vit-base-patch16-224_embeddings_softmax.pkl",
        neural_data_path="data/processed/hybrid_neural_responses.npy"
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for batch in loader:
        print(batch["image_embedding"].shape)  # (64, D)
        print(batch["neuron_idx"].shape)       # (64,)
        print(batch["response"].shape)         # (64,)
        break
import os
import numpy as np
import pickle

input_dir = "/home/maria/Documents/AllenBrainObservatory/neural_activity_matrices"
output_dir = "/home/maria/LuckyMouse/src/data"
os.makedirs(output_dir, exist_ok=True)

output_npy_path = os.path.join(output_dir, "hybrid_neural_responses.npy")
output_pkl_path = os.path.join(output_dir, "hybrid_neural_responses_index.pkl")

all_data = []
index_map = {}
current_row = 0

for filename in sorted(os.listdir(input_dir)):
    if filename.endswith("_neural_responses.npy"):
        filepath = os.path.join(input_dir, filename)
        data = np.load(filepath)
        n_rows = data.shape[0]

        prefix = filename.split("_")[0]
        index_map[prefix] = (current_row, current_row + n_rows)
        current_row += n_rows

        all_data.append(data)

hybrid_data = np.vstack(all_data)

np.save(output_npy_path, hybrid_data)
with open(output_pkl_path, "wb") as f:
    pickle.dump(index_map, f)

print(f"Saved hybrid data: {output_npy_path}")
print(f"Saved index map: {output_pkl_path}")

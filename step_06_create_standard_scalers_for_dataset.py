import os
import json
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# === Configuration Paths ===
# Defines the base directory of the processed dataset.
# This path points to the specific timestamped output folder from the cleaning step.
# Example: "processed_datasets/dataset_combined_mixed_20250429_193157"
BASE_DIR = os.path.join("processed_datasets", "dataset_combined_mixed_YYYYMMDD_HHMMSS") # <--- MANUAL UPDATE REQUIRED

# Defines the path to the training JSONL file.
JSONL_TRAIN_PATH = os.path.join(BASE_DIR, "dataset_train.jsonl")
# Defines the directory containing the density matrix (.npz) files.
DM_DIR = os.path.join(BASE_DIR, "dm")
# Defines the output directory where the fitted scalers will be saved.
SCALER_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "scalers")

# Creates the output directory for scalers if it does not exist.
os.makedirs(SCALER_OUTPUT_DIR, exist_ok=True)

# === Configuration Parameters ===
# Defines the batch size for `partial_fit` operations on large datasets.
BATCH_SIZE = 100

# === Scaler Initialization ===
# Initializes StandardScaler objects for each feature type that requires scaling.
scaler_real = StandardScaler()
scaler_imag = StandardScaler()
scaler_classical = StandardScaler()
scaler_target = StandardScaler()
scaler_diag_real = StandardScaler()
scaler_diag_imag = StandardScaler()

# === Data Buffers ===
# Buffers to temporarily store data before `partial_fit` calls.
real_buffer = []
imag_buffer = []
diag_real_buffer = []
diag_imag_buffer = []
classical_buffer = []
target_buffer = []

# Counters for tracking processing status.
missing_files_count = 0
total_valid_samples = 0

# === Precompute Density Matrix Dimension ===
try:
    # Loads an example density matrix to determine its dimension.
    # Assumes at least one .npz file exists in the DM_DIR.
    example_dm_path = os.path.join(DM_DIR, os.listdir(DM_DIR)[0])
    example_dm_vec = np.load(example_dm_path)["dm"]
    # L is the length of the real/imaginary vector part of the flattened DM.
    L = len(example_dm_vec) // 2
    # N is the dimension of the square density matrix (N x N).
    N = int((np.sqrt(1 + 8 * L) - 1) // 2)
except Exception as e:
    raise RuntimeError(f"Failed to determine density matrix size from {DM_DIR}. Ensure .npz files exist and are valid.") from e

# === Helper Function ===
def extract_classical_features(entry: dict) -> list:
    """
    Extracts numerical classical features from a dataset entry's circuit features.
    """
    features = entry["input"]["circuit_features"]
    # Retrieves gate counts and circuit depth, defaulting to 0 if not present.
    cx_gates = features.get("gate_counts", {}).get("cx", 0)
    u_gates = features.get("gate_counts", {}).get("u", 0)
    circuit_depth = features.get("depth", 0)
    return [circuit_depth, cx_gates, u_gates]

# === Step 1: Stream Training Data and Partially Fit Scalers ===
print("\nStep 1: Streaming training data and incrementally fitting scalers...")
with open(JSONL_TRAIN_PATH, "r") as f:
    for line in tqdm(f, desc="Processing training entries"):
        item = json.loads(line)
        # Extracts the core data entry, handling cases where it's nested under "jsonl" key.
        entry = item.get("jsonl", item)

        # Constructs the full path to the density matrix file.
        dm_file_path = os.path.join(DM_DIR, os.path.basename(entry["input"]["density_matrix_path"]))
        
        # Skips entry if the corresponding .npz file does not exist.
        if not os.path.exists(dm_file_path):
            missing_files_count += 1
            continue

        try:
            # Loads the density matrix vector from the .npz file.
            dm_vector = np.load(dm_file_path)["dm"]
        except Exception as e:
            print(f"Skipping corrupt or unreadable .npz file: {dm_file_path} ({str(e)})")
            missing_files_count += 1
            continue

        # Separates the real and imaginary parts of the flattened density matrix vector.
        real_vector = dm_vector[:L]
        imag_vector = dm_vector[L:]

        # Appends data to respective buffers.
        real_buffer.append(real_vector)
        imag_buffer.append(imag_vector)
        classical_buffer.append(extract_classical_features(entry))
        target_buffer.append(entry["target"]["delta_trace"])

        # Reconstructs the full density matrix to extract its diagonal elements.
        real_matrix = np.zeros((N, N), dtype=np.float32)
        imag_matrix = np.zeros((N, N), dtype=np.float32)
        idx = 0
        for row in range(N):
            for col in range(row, N):
                real_matrix[row, col] = real_vector[idx]
                imag_matrix[row, col] = imag_vector[idx]
                if row != col:
                    real_matrix[col, row] = real_vector[idx]
                    imag_matrix[col, row] = -imag_vector[idx] # Imaginary part is anti-symmetric
                idx += 1

        diag_real_buffer.append(real_matrix.diagonal().copy())
        diag_imag_buffer.append(imag_matrix.diagonal().copy())
        total_valid_samples += 1

        # Performs partial fitting when buffers reach the defined batch size.
        if len(real_buffer) >= BATCH_SIZE:
            scaler_real.partial_fit(np.vstack(real_buffer).reshape(-1, 1))
            scaler_imag.partial_fit(np.vstack(imag_buffer).reshape(-1, 1))
            scaler_diag_real.partial_fit(np.vstack(diag_real_buffer))
            scaler_diag_imag.partial_fit(np.vstack(diag_imag_buffer))
            # Clears buffers after partial fit.
            real_buffer.clear()
            imag_buffer.clear()
            diag_real_buffer.clear()
            diag_imag_buffer.clear()

# Performs final partial fitting for any remaining data in buffers.
if real_buffer:
    scaler_real.partial_fit(np.vstack(real_buffer).reshape(-1, 1))
    scaler_imag.partial_fit(np.vstack(imag_buffer).reshape(-1, 1))
    scaler_diag_real.partial_fit(np.vstack(diag_real_buffer))
    scaler_diag_imag.partial_fit(np.vstack(diag_imag_buffer))

# === Step 2: Fit Classical and Target Scalers ===
print("\nStep 2: Fitting classical features and target (delta_trace) scalers...")
# Fits classical features scaler.
scaler_classical.fit(np.array(classical_buffer))
# Fits target variable scaler. Reshaping to (-1, 1) is necessary for StandardScaler single-feature input.
scaler_target.fit(np.array(target_buffer).reshape(-1, 1))

# === Step 3: Save Scalers ===
print("\nStep 3: Saving fitted scalers to disk...")
# Consolidates all fitted scalers into a dictionary for easy saving.
scalers_to_save = {
    "scaler_real": scaler_real,
    "scaler_imag": scaler_imag,
    "scaler_classical": scaler_classical,
    "scaler_target": scaler_target,
    "scaler_diag_real": scaler_diag_real,
    "scaler_diag_imag": scaler_diag_imag,
}

# Saves each scaler object using Python's pickle module.
for name, scaler in tqdm(scalers_to_save.items(), desc="Saving scalers"):
    with open(os.path.join(SCALER_OUTPUT_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(scaler, f)

# === Summary ===
print(f"\nScaling process complete! Fitted scalers saved to: {SCALER_OUTPUT_DIR}")
print(f"Skipped {missing_files_count} entries due to missing or corrupt .npz files.")
print(f"Used {total_valid_samples} valid samples for scaler fitting.")
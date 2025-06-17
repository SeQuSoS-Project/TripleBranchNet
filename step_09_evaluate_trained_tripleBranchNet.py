import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
# Import CrosstalkDataset and TripleBranchNet from the training script as instructed.
# Ensure that 'step_07_train_tripleBranchNet_model.py' is in your Python path or the same directory.
from step_07_train_tripleBranchNet_model import CrosstalkDataset, TripleBranchNet 

# === Reproducibility ===
# Sets random seeds for NumPy and PyTorch to ensure consistent results.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True # Ensures deterministic behavior for CUDA (if used)
torch.backends.cudnn.benchmark = False # Disables benchmarking for reproducibility

# Required for loading models that contain custom classes when weights_only=False is used.
# This registers the custom class (TripleBranchNet) to be safely loaded by PyTorch.
torch.serialization.add_safe_globals([TripleBranchNet])

# === Configuration Paths ===
# Defines the base directory of the processed dataset.
# This path points to the specific timestamped output folder from the cleaning step.
# Example: "processed_datasets/dataset_combined_mixed_20250429_193157"
BASE_DATASET_DIR = os.path.join("processed_datasets", "dataset_combined_mixed_YYYYMMDD_HHMMSS") # <--- MANUAL UPDATE REQUIRED

# Defines the timestamp of the specific training run to evaluate.
# This timestamp corresponds to the `output/YYYYMMDD_HHMMSS` folder created during training in step_07_train_tripleBranchNet_model.py.
TIMESTAMP_TO_EVALUATE = "YYYYMMDD_HHMMSS" # <--- MANUAL UPDATE REQUIRED

# Constructs the path to the subdirectory containing models and scalers for the chosen run.
SUB_RUN_DIR = os.path.join(BASE_DATASET_DIR, "output", TIMESTAMP_TO_EVALUATE)

# Defines paths for loading the target scaler and the trained models.
SCALER_TARGET_PATH = os.path.join(SUB_RUN_DIR, "scalers", "scaler_target.pkl")
MODEL_BEST_PATH = os.path.join(SUB_RUN_DIR, "model", "model_best_full.pt")
MODEL_LAST_PATH = os.path.join(SUB_RUN_DIR, "model", "model_last_full.pt")
# Defines the path to the test dataset JSONL file.
TEST_JSONL_PATH = os.path.join(BASE_DATASET_DIR, "dataset_test.jsonl") # Retained for clarity, though not directly used in DataLoader init

# Defines the output directory for evaluation results (CSV, plots).
EVALUATION_OUTPUT_DIR = os.path.join(SUB_RUN_DIR, "test_evaluation")
# Creates the output directory if it does not exist.
os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

# === Load Target Scaler ===
print("Loading target scaler...")
with open(SCALER_TARGET_PATH, "rb") as f:
    scaler_target = pickle.load(f)
print(" Target scaler loaded.")

# === Load Test Dataset ===
print("Loading test dataset...")
test_dataset = CrosstalkDataset(BASE_DATASET_DIR, split_jsonl="dataset_test.jsonl")
# Configures DataLoader for the test dataset.
TEST_BATCH_SIZE = 8
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
print(f" Test dataset loaded with {len(test_dataset)} samples.")

# === Load Trained Models ===
# Determines the device for model inference (GPU if available, otherwise CPU).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading models to device: {DEVICE}")

# Loads the 'best' and 'last' trained models. `map_location` ensures loading to correct device.
# `weights_only=False` requires the model's class definition to be available in scope (via import).
model_best = torch.load(MODEL_BEST_PATH, map_location=DEVICE).eval() # Sets model to evaluation mode
model_last = torch.load(MODEL_LAST_PATH, map_location=DEVICE).eval() # Sets model to evaluation mode
print(" Models loaded successfully.")

# === Inference Loop ===
print("Running inference on the test set...")
# Stores results for later analysis.
evaluation_results = []
for batch in tqdm(test_loader, desc="Evaluating Models"):
    # Moves batch data to the specified device.
    quantum_input = batch["quantum"].to(DEVICE)
    classical_input = batch["classical"].to(DEVICE)
    diagonal_input = batch["diag"].to(DEVICE)
    true_target_scaled = batch["target"].cpu().numpy() # True target values (scaled)

    # Performs inference without gradient tracking.
    with torch.no_grad():
        predicted_best_scaled = model_best(quantum_input, classical_input, diagonal_input).cpu().numpy()
        predicted_last_scaled = model_last(quantum_input, classical_input, diagonal_input).cpu().numpy()

    # Rescales predictions and true values back to their original data range.
    predicted_best_rescaled = scaler_target.inverse_transform(predicted_best_scaled.reshape(-1, 1)).flatten()
    predicted_last_rescaled = scaler_target.inverse_transform(predicted_last_scaled.reshape(-1, 1)).flatten()
    true_rescaled = scaler_target.inverse_transform(true_target_scaled.reshape(-1, 1)).flatten()

    # Appends batch results to the list.
    for p_best, p_last, t_val in zip(predicted_best_rescaled, predicted_last_rescaled, true_rescaled):
        evaluation_results.append({
            "predicted_best": p_best,
            "predicted_last": p_last,
            "true": t_val,
            "error_best": p_best - t_val,
            "error_last": p_last - t_val
        })

# Converts results to a Pandas DataFrame for easy analysis and saving.
df_results = pd.DataFrame(evaluation_results)

# === Save Results to CSV ===
RESULTS_CSV_PATH = os.path.join(EVALUATION_OUTPUT_DIR, "test_predictions_comparison.csv")
df_results.to_csv(RESULTS_CSV_PATH, index=False)
print(f" Saved prediction comparison to: {RESULTS_CSV_PATH}")

# === Generate Scatter Plots (Predicted vs True) ===
plt.figure(figsize=(12, 6)) # Adjust figure size for better side-by-side view

# Plot for the "Best" model.
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
plt.scatter(df_results["true"], df_results["predicted_best"], alpha=0.5, s=10)
plt.xlabel("True Delta Trace")
plt.ylabel("Predicted Delta Trace (Best Model)")
plt.title("Best Model: Predicted vs True")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box') # Ensures equal aspect ratio
min_val = min(df_results["true"].min(), df_results["predicted_best"].min())
max_val = max(df_results["true"].max(), df_results["predicted_best"].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1) # Identity line

# Plot for the "Last" model.
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
plt.scatter(df_results["true"], df_results["predicted_last"], alpha=0.5, s=10, color="orange")
plt.xlabel("True Delta Trace")
plt.ylabel("Predicted Delta Trace (Last Model)")
plt.title("Last Model: Predicted vs True")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
min_val = min(df_results["true"].min(), df_results["predicted_last"].min())
max_val = max(df_results["true"].max(), df_results["predicted_last"].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1) # Identity line

plt.tight_layout() # Adjusts plot parameters for a tight layout.
SCATTER_PLOT_PATH = os.path.join(EVALUATION_OUTPUT_DIR, "scatter_comparison.pdf")
plt.savefig(SCATTER_PLOT_PATH)
plt.close() # Closes the figure to free memory.
print(f" Saved scatter plot comparison to: {SCATTER_PLOT_PATH}")

# === Generate Residual Histograms ===
plt.figure(figsize=(10, 6))
# Histogram for "Best" model residuals.
plt.hist(df_results["error_best"], bins=50, alpha=0.7, label="Best Model Residuals", color="blue", density=False)
# Histogram for "Last" model residuals.
plt.hist(df_results["error_last"], bins=50, alpha=0.7, label="Last Model Residuals", color="orange", density=False)
plt.xlabel("Prediction Error (True - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors (Residuals)")
plt.legend()
plt.grid(True)
plt.tight_layout()
RESIDUALS_PLOT_PATH = os.path.join(EVALUATION_OUTPUT_DIR, "residuals_comparison.pdf")
plt.savefig(RESIDUALS_PLOT_PATH)
plt.close()
print(f" Saved residual histogram comparison to: {RESIDUALS_PLOT_PATH}")

# === Generate Error vs True Value Scatter Plot ===
plt.figure(figsize=(12, 6))
plt.scatter(df_results["true"], df_results["error_best"], alpha=0.5, s=10, label="Best Model Error", color="blue")
plt.scatter(df_results["true"], df_results["error_last"], alpha=0.5, s=10, label="Last Model Error", color="orange")
plt.axhline(0, linestyle="--", color="black", linewidth=1.5) # Horizontal line at zero error
plt.xlabel("True Delta Trace")
plt.ylabel("Prediction Error (True - Predicted)")
plt.title("Prediction Error vs True Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
ERROR_VS_TRUE_PLOT_PATH = os.path.join(EVALUATION_OUTPUT_DIR, "error_vs_true.pdf")
plt.savefig(ERROR_VS_TRUE_PLOT_PATH)
plt.close()
print(f" Saved error vs true value plot to: {ERROR_VS_TRUE_PLOT_PATH}")

# === Compute and Log Metrics ===
# Calculates MAE, RMSE, and R2 for both "best" and "last" models.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Re-import here for clarity as used directly
evaluation_metrics = {
    "best_model": {
        "MAE": float(mean_absolute_error(df_results["true"], df_results["predicted_best"])),
        "RMSE": float(np.sqrt(mean_squared_error(df_results["true"], df_results["predicted_best"]))),
        "R2": float(r2_score(df_results["true"], df_results["predicted_best"]))
    },
    "last_model": {
        "MAE": float(mean_absolute_error(df_results["true"], df_results["predicted_last"])),
        "RMSE": float(np.sqrt(mean_squared_error(df_results["true"], df_results["predicted_last"]))),
        "R2": float(r2_score(df_results["true"], df_results["predicted_last"]))
    }
}

# Prints the computed metrics to console.
for tag, metrics_data in evaluation_metrics.items():
    print(f"\n--- {tag.replace('_', ' ').upper()} ---")
    print(f"  MAE  : {metrics_data['MAE']:.6f}")
    print(f"  RMSE : {metrics_data['RMSE']:.6f}")
    print(f"  R²   : {metrics_data['R2']:.4f}")

# === Save to JSON ===
METRICS_JSON_PATH = os.path.join(EVALUATION_OUTPUT_DIR, "test_metrics_summary.json")
with open(METRICS_JSON_PATH, "w") as f:
    json.dump(evaluation_metrics, f, indent=2)
print(f" Saved evaluation metrics summary to: {METRICS_JSON_PATH}")

print("\n Model evaluation complete!")
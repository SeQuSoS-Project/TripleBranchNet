import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader # Included Dataset here
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from datetime import datetime
from collections import deque
import multiprocessing
import pickle # Added for dataset class

# === Seed for Reproducibility ===
# Sets random seeds across various libraries (Python, NumPy, PyTorch)
# to ensure training and data loading processes are reproducible.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Sets CUDA seeds if a GPU is available.
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# === Configuration Paths ===
# Defines the base directory of the processed dataset,
# which is the output from the cleaning step.
# IMPORTANT: Update 'dataset_combined_mixed_YYYYMMDD_HHMMSS' to match the exact
# timestamped folder name from your cleaned dataset.
BASE_DATASET_DIR = os.path.join("processed_datasets", "dataset_combined_mixed_YYYYMMDD_HHMMSS") # <--- MANUAL UPDATE REQUIRED

# Creates a timestamped subfolder for the current training run's outputs.
CURRENT_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = os.path.join(BASE_DATASET_DIR, "output", CURRENT_RUN_TIMESTAMP)

# Defines specific output directories for the trained model and various logs.
MODEL_SAVE_DIR = os.path.join(RUN_OUTPUT_DIR, "model")
LOG_ROOT_DIR = os.path.join(RUN_OUTPUT_DIR, "logs")
LOG_SUBDIRS = {
    "summary": os.path.join(LOG_ROOT_DIR, "summary"),        # Stores per-epoch summary JSONs
    "curves": os.path.join(LOG_ROOT_DIR, "curves"),          # Stores training curves (CSV, plots)
    "predictions": os.path.join(LOG_ROOT_DIR, "predictions"), # Stores scatter plots of predictions
    "residuals": os.path.join(LOG_ROOT_DIR, "residuals"),     # Stores histograms of residuals
}

# Creates all necessary output directories if they do not exist.
for directory in LOG_SUBDIRS.values():
    os.makedirs(directory, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Defines the path for saving the best performing model's state dictionary.
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "model_best.pt")

# === Training Hyperparameters ===
BATCH_SIZE = 16 # Number of samples per training batch.
EPOCHS = 50 # Total number of training epochs.
LEARNING_RATE = 3e-4 # Initial learning rate for the optimizer.
WEIGHT_DECAY = 1e-4 # L2 regularization coefficient.
WARMUP_EPOCHS = 5 # Number of epochs for linear learning rate warmup.
EARLY_STOP_PATIENCE = None # Number of epochs to wait for improvement before stopping (None = no early stopping).
# Determines the device for training (GPU if available, otherwise CPU).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Noise Augmentation Parameters ===
# Controls whether noise augmentation is applied to quantum input data during training.
USE_NOISE = False
NOISE_STD = 0.005 # Standard deviation of Gaussian noise when augmentation is enabled.
# Schedule for noise application (e.g., "constant", "linear_decay", "cosine_decay").
NOISE_SCHEDULE = "constant"
# Defines the min/max bounds for clamping quantum data after noise augmentation.
QUANTUM_DATA_MIN = -5.0
QUANTUM_DATA_MAX = 5.0


# --- CrosstalkDataset Class ---
# This class extends PyTorch's Dataset to load, preprocess, and scale the data.
class CrosstalkDataset(Dataset):
    def __init__(self, base_directory: str, split_jsonl: str = "dataset.jsonl"):
        """
        Initializes the dataset, loading entry metadata and pre-fitted scalers.

        Args:
            base_directory: The root directory containing 'dm/', 'output/scalers/',
                            and the split JSONL files.
            split_jsonl: The filename of the JSONL split (e.g., "dataset_train.jsonl").
        """
        self.base_dir = base_directory
        self.dm_dir = os.path.join(base_directory, "dm")
        self.jsonl_path = os.path.join(base_directory, split_jsonl)
        self.scaler_dir = os.path.join(base_directory, "output", "scalers")

        # Loads all dataset entries from the specified JSONL file.
        self.entries = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                # Extracts the core data entry, handling potential nesting.
                entry = item.get("jsonl", item)
                self.entries.append(entry)

        # Loads the pre-fitted StandardScaler objects from disk.
        self.scaler_real = self._load_scaler("scaler_real.pkl")
        self.scaler_imag = self._load_scaler("scaler_imag.pkl")
        self.scaler_classical = self._load_scaler("scaler_classical.pkl")
        self.scaler_target = self._load_scaler("scaler_target.pkl")
        self.scaler_diag_real = self._load_scaler("scaler_diag_real.pkl")
        self.scaler_diag_imag = self._load_scaler("scaler_diag_imag.pkl")

    def _load_scaler(self, filename: str):
        """Helper to load a StandardScaler from a pickle file."""
        path = os.path.join(self.scaler_dir, filename)
        with open(path, "rb") as f:
            return pickle.load(f)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves and preprocesses a single data sample at the given index.
        This includes loading, scaling, and formatting data for model input.
        """
        entry = self.entries[idx]

        # === Load and Scale Density Matrix Components ===
        # Constructs the path to the density matrix NPZ file.
        dm_path = os.path.join(self.dm_dir, os.path.basename(entry["input"]["density_matrix_path"]))
        dm_vec = np.load(dm_path)["dm"]
        
        # Splits the flattened density matrix vector into real and imaginary parts.
        L = len(dm_vec) // 2 # Length of one part
        real_vec = dm_vec[:L]
        imag_vec = dm_vec[L:]

        # Scales the real and imaginary components using their respective scalers.
        real_scaled = self.scaler_real.transform(real_vec.reshape(-1, 1)).flatten()
        imag_scaled = self.scaler_imag.transform(imag_vec.reshape(-1, 1)).flatten()

        # Reconstructs the full square density matrix from the scaled flattened vectors.
        N = int((np.sqrt(1 + 8 * L) - 1) // 2) # Original matrix dimension
        real_mat = np.zeros((N, N), dtype=np.float32)
        imag_mat = np.zeros((N, N), dtype=np.float32)

        i = 0
        for row in range(N):
            for col in range(row, N):
                real_mat[row, col] = real_scaled[i]
                imag_mat[row, col] = imag_scaled[i]
                if row != col: # Populates the lower triangle based on Hermitian properties
                    real_mat[col, row] = real_scaled[i]
                    imag_mat[col, row] = -imag_scaled[i]
                i += 1
        
        # Stacks real and imaginary matrices for CNN input (channels-first format).
        stacked_quantum_input = np.stack([real_mat, imag_mat])
        quantum_tensor = torch.from_numpy(stacked_quantum_input).float()

        # === Extract and Scale Diagonal Components ===
        # Extracts diagonal elements from the reconstructed matrices.
        diag_real = np.diag(real_mat).reshape(1, -1)
        diag_imag = np.diag(imag_mat).reshape(1, -1)
        # Scales the diagonal components.
        diag_real_scaled = self.scaler_diag_real.transform(diag_real).flatten()
        diag_imag_scaled = self.scaler_diag_imag.transform(diag_imag).flatten()
        # Stacks scaled diagonals for MLP input.
        diag_tensor = torch.from_numpy(np.stack([diag_real_scaled, diag_imag_scaled])).float()

        # === Extract and Scale Classical Features ===
        features = entry["input"]["circuit_features"]
        # Retrieves classical features, providing default values for robustness.
        circuit_depth = features.get("depth", 0)
        cx_gates = features.get("gate_counts", {}).get("cx", 0)
        u_gates = features.get("gate_counts", {}).get("u", 0)
        
        classical_raw = np.array([[circuit_depth, cx_gates, u_gates]], dtype=np.float32)
        # Scales classical features.
        classical_scaled = self.scaler_classical.transform(classical_raw).flatten()
        classical_tensor = torch.tensor(classical_scaled, dtype=torch.float32)

        # === Scale Target Variable ===
        target_value = entry["target"]["delta_trace"]
        # Scales the target value.
        target_scaled = self.scaler_target.transform([[target_value]])[0][0]
        target_tensor = torch.tensor(target_scaled, dtype=torch.float32)

        # Returns a dictionary of preprocessed and scaled tensors.
        return {
            "quantum": quantum_tensor,    # Shape: [2, N, N]
            "classical": classical_tensor,# Shape: [3]
            "diag": diag_tensor,          # Shape: [2, N]
            "target": target_tensor       # Scalar
        }

# === Worker Initialization Function ===
def worker_init_fn(worker_id: int):
    """
    Initializes random seeds for each DataLoader worker process.
    Ensures data loading is reproducible across multiple workers.
    """
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

# === Global Dataset Variables (initialized in main process) ===
# These will hold the loaded dataset objects.
train_ds = None
val_ds = None

def load_datasets():
    """
    Loads training and validation datasets using the CrosstalkDataset class.
    This function is intended to be called only once in the main process.
    Includes error handling for missing files.
    """
    global train_ds, val_ds # Modifies global variables
    try:
        train_ds = CrosstalkDataset(BASE_DATASET_DIR, split_jsonl="dataset_train.jsonl")
        val_ds = CrosstalkDataset(BASE_DATASET_DIR, split_jsonl="dataset_val.jsonl")
        print(f" Successfully loaded datasets from: {BASE_DATASET_DIR}")
        print(f"   Train samples: {len(train_ds)}")
        print(f"   Validation samples: {len(val_ds)}")
        
        # Displays a sample of the target range for verification.
        sample_targets = [train_ds[i]['target'].item() for i in range(min(100, len(train_ds)))]
        print(f"   Sampled target range (scaled): [{min(sample_targets):.4f}, {max(sample_targets):.4f}]")
        print(f"   Sampled target mean (scaled): {np.mean(sample_targets):.4f}, std: {np.std(sample_targets):.4f}")
        
    except FileNotFoundError as e:
        print(f" Error: Dataset files not found. Check BASE_DATASET_DIR configuration.")
        print(f"   Details: {e}")
        raise # Re-raises the exception after printing
    except Exception as e:
        print(f" An unexpected error occurred while loading datasets: {e}")
        print(f"   Base directory attempted: {BASE_DATASET_DIR}")
        raise

# === Triple-Branch Neural Network Model ===
class TripleBranchNet(nn.Module):
    """
    A multi-branch neural network designed to process quantum, classical,
    and diagonal features independently before fusion for final prediction.
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        
        # Quantum branch (CNN): Processes the 2D density matrix (real and imaginary parts).
        self.qcnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(), # Swish activation function
            nn.Dropout2d(dropout), # Dropout for regularization
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            
            nn.AdaptiveAvgPool2d((4, 4)) # Adaptive pooling to a fixed output size
        )

        # Classical branch (MLP): Processes classical circuit features (e.g., depth, gate counts).
        self.classical_mlp = nn.Sequential(
            nn.Linear(3, 32), # Input dimension 3
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 32), # Output dimension 32
            nn.LayerNorm(32),
            nn.SiLU()
        )

        # Diagonal branch (MLP): Processes the diagonal elements of the density matrix.
        # Input dimension (2 * N), where N=1024, so 2048.
        self.diag_mlp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64), # Output dimension 64
            nn.SiLU()
        )

        # LayerNormalization layers for stabilizing outputs of individual branches before fusion.
        self.norm_q = nn.LayerNorm(512 * 4 * 4) # Flattened output size of qcnn
        self.norm_c = nn.LayerNorm(32) # Output size of classical_mlp
        self.norm_d = nn.LayerNorm(64) # Output size of diag_mlp

        # Fusion MLP: Combines processed features from all three branches.
        # Input dimension is the sum of output dimensions from the normalized branches.
        FUSION_INPUT_DIM = 512 * 4 * 4 + 32 + 64
        self.fusion = nn.Sequential(
            nn.Linear(FUSION_INPUT_DIM, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 1) # Final output layer for regression (single scalar prediction)
        )

    def forward(self, quantum: torch.Tensor, classical: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            quantum: Tensor for quantum input (density matrix image).
            classical: Tensor for classical circuit features.
            diag: Tensor for density matrix diagonal features.

        Returns:
            The model's scalar prediction for delta_trace.
        """
        # Process quantum input through CNN branch and flatten its output.
        q = self.qcnn(quantum)
        q = q.view(q.size(0), -1) # Flatten for MLP input
        q = self.norm_q(q) # Apply LayerNormalization

        # Process classical input through MLP branch.
        c = self.classical_mlp(classical)
        c = self.norm_c(c) # Apply LayerNormalization

        # Process diagonal input through MLP branch.
        d = self.diag_mlp(diag.view(diag.size(0), -1)) # Flatten diag input if necessary
        d = self.norm_d(d) # Apply LayerNormalization

        # Concatenate outputs from all three branches.
        x = torch.cat([q, c, d], dim=1)
        # Pass the concatenated features through the fusion MLP for final prediction.
        out = self.fusion(x)
        out = out.squeeze(-1) # Removes the last dimension if it's 1 (e.g., [batch_size, 1] -> [batch_size])
        
        return out

# === Metrics Accumulator Class ===
class MetricsAccumulator:
    """
    Accumulates true and predicted values over batches to compute comprehensive
    regression metrics at the end of an epoch or evaluation phase.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Resets all accumulated data and metrics sums."""
        self.y_true_all = []
        self.y_pred_all = []
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.count = 0 # Total number of samples processed

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Updates the accumulator with true and predicted values from a batch.

        Args:
            y_true: Ground truth tensor.
            y_pred: Predicted values tensor.
        """
        # Detaches tensors from computation graph and moves to CPU for NumPy conversion.
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        
        self.y_true_all.extend(y_true_np)
        self.y_pred_all.extend(y_pred_np)
        
        batch_size = len(y_true_np)
        self.mse_sum += np.mean((y_true_np - y_pred_np) ** 2) * batch_size
        self.mae_sum += np.mean(np.abs(y_true_np - y_pred_np)) * batch_size
        self.count += batch_size
    
    def compute(self) -> dict:
        """
        Computes and returns a dictionary of all aggregated metrics.
        Includes MSE, RMSE, MAE, R², Spearman/Pearson correlations, SMAPE, and error percentiles.
        """
        y_true = np.array(self.y_true_all)
        y_pred = np.array(self.y_pred_all)
        
        residuals = y_true - y_pred # Calculate residuals

        # Basic regression metrics.
        mse = self.mse_sum / self.count
        rmse = np.sqrt(mse)
        mae = self.mae_sum / self.count
        
        r2 = r2_score(y_true, y_pred) # R² score
        residual_std = np.std(residuals) # Standard deviation of residuals
        
        # Symmetric Mean Absolute Percentage Error (SMAPE).
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        # Avoids division by zero for SMAPE calculation.
        non_zero_mask = denominator > 1e-8
        if np.any(non_zero_mask): # Use np.any for numpy array
            smape = np.mean(np.abs(residuals[non_zero_mask]) / denominator[non_zero_mask]) * 100
        else:
            smape = 0.0
        
        # Error percentiles.
        max_error = np.max(np.abs(residuals))
        p90_error = np.percentile(np.abs(residuals), 90)
        p95_error = np.percentile(np.abs(residuals), 95)
        
        # Correlation metrics.
        try:
            spearman_corr, _ = spearmanr(y_true, y_pred)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        except Exception:
            spearman_corr = 0.0
            
        try:
            pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
        except Exception:
            pearson_corr = 0.0
        
        return {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'residual_std': residual_std, 'smape': smape,
            'max_error': max_error, 'p90_error': p90_error, 'p95_error': p95_error,
            'pearson_corr': pearson_corr, 'spearman_corr': spearman_corr,
            'y_true': y_true, 'y_pred': y_pred # Include for plotting/analysis outside
        }

# === Training Loop Function ===
def train_model():
    """
    Executes the full training and validation loop for the TripleBranchNet model.
    Handles data loading, model initialization, optimization, learning rate scheduling,
    metric computation, logging, and model saving.
    """
    global train_ds, val_ds # Access global dataset variables
    
    # Loads datasets if not already loaded (ensures this runs only once in main process).
    if train_ds is None or val_ds is None:
        load_datasets()
    
    # Configures DataLoader for training and validation datasets.
    # Adjusts `num_workers` and `multiprocessing_context` for Windows compatibility.
    if os.name == 'nt': # Checks if running on Windows
        num_workers = 0 # No multiprocessing for data loading on Windows due to common issues
        persistent_workers = False
    else: # Linux/macOS
        num_workers = 4 # Enables multiprocessing for faster data loading
        persistent_workers = True # Keeps worker processes alive between epochs
        
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, # Shuffles training data each epoch
        num_workers=num_workers,
        worker_init_fn=worker_init_fn, # Initializes worker seeds for reproducibility
        persistent_workers=persistent_workers,
        # 'spawn' context is safer for multiprocessing in PyTorch
        multiprocessing_context='spawn' if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # No shuffling for validation data
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        multiprocessing_context='spawn' if num_workers > 0 else None
    )
    
    # Initializes the model, optimizer, and learning rate scheduler.
    model = TripleBranchNet(dropout=0.1).to(DEVICE) # Moves model to the configured device
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Defines a sequential learning rate scheduler: linear warmup followed by cosine annealing.
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS - WARMUP_EPOCHS, # Cosine annealing for remaining epochs after warmup
        eta_min=1e-6 # Minimum learning rate
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[WARMUP_EPOCHS] # Switches from warmup to cosine at this epoch
    )
    
    loss_fn = nn.MSELoss() # Mean Squared Error loss function for regression

    best_val_loss = float("inf") # Tracks the best validation loss achieved
    best_epoch = 0 # Records the epoch at which the best model was saved
    patience_counter = 0 # Counter for early stopping patience
    
    # Lists to store metrics and loss values across epochs for logging and plotting.
    train_losses_history, val_losses_history, lrs_history, epochs_history = [], [], [], []
    all_validation_metrics = [] # Stores full metric dictionaries for each epoch
    
    global_step = 0 # Tracks total training steps (batches processed)

    # === Main Training Loop ===
    for epoch in range(1, EPOCHS + 1):
        # Sets the model to training mode.
        model.train()
        total_train_loss = 0.0
        # Configures progress bar for training, disabled for non-main multiprocessing workers.
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False, 
                                 disable=multiprocessing.current_process().name != 'MainProcess')

        for batch in train_progress_bar:
            q_input = batch["quantum"].to(DEVICE)
            
            # Applies optional noise augmentation to quantum input data.
            if USE_NOISE and model.training: # Only apply noise during training phase
                noise_scale = 0.0 # Default noise scale
                if NOISE_SCHEDULE == "constant":
                    noise_scale = NOISE_STD
                elif NOISE_SCHEDULE == "linear_decay":
                    # Noise decreases linearly over epochs.
                    noise_scale = NOISE_STD * (1 - (epoch - 1) / EPOCHS)
                elif NOISE_SCHEDULE == "cosine_decay":
                    # Noise decreases following a cosine curve.
                    noise_scale = NOISE_STD * 0.5 * (1 + np.cos(np.pi * (epoch - 1) / EPOCHS))
                
                noise = torch.randn_like(q_input) * noise_scale
                q_input = q_input + noise
                
                # Clips noisy quantum data to stay within defined bounds.
                q_input = torch.clamp(q_input, min=QUANTUM_DATA_MIN, max=QUANTUM_DATA_MAX)
            
            c_input = batch["classical"].to(DEVICE)
            d_input = batch["diag"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            optimizer.zero_grad() # Clears previous gradients.
            output = model(q_input, c_input, d_input) # Performs forward pass.
            loss = loss_fn(output, target) # Calculates loss.
            
            # Checks for NaN or Inf loss values, skipping problematic batches.
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Training loss is {loss.item()} at step {global_step}. Skipping batch.")
                continue
                
            loss.backward() # Computes gradients.
            optimizer.step() # Updates model weights.
            global_step += 1

            batch_loss = loss.item()
            total_train_loss += batch_loss * q_input.size(0)
            train_progress_bar.set_postfix({"loss": batch_loss}) # Updates progress bar with current loss

        avg_train_loss = total_train_loss / len(train_ds) # Calculates average training loss for the epoch.

        # === Validation Phase ===
        # Sets the model to evaluation mode (disables dropout, batch norm updates).
        model.eval()
        total_val_loss = 0.0
        metrics_accumulator = MetricsAccumulator() # Initializes accumulator for validation metrics.
        # Configures progress bar for validation.
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]", leave=False,
                                disable=multiprocessing.current_process().name != 'MainProcess')
        
        with torch.no_grad(): # Disables gradient calculations for efficiency.
            for batch in val_progress_bar:
                q_input = batch["quantum"].to(DEVICE)
                c_input = batch["classical"].to(DEVICE)
                d_input = batch["diag"].to(DEVICE)
                target = batch["target"].to(DEVICE)

                output = model(q_input, c_input, d_input)
                loss = loss_fn(output, target)
                
                # Skips batch if validation loss is NaN or Inf.
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Validation loss is {loss.item()} in batch. Skipping batch.")
                    continue
                    
                total_val_loss += loss.item() * q_input.size(0)
                metrics_accumulator.update(target, output) # Updates metrics for evaluation.

        avg_val_loss = total_val_loss / len(val_ds) # Calculates average validation loss.
        scheduler.step() # Steps the learning rate scheduler.
        current_lr = optimizer.param_groups[0]['lr'] # Gets the current learning rate.

        # Computes and stores comprehensive validation metrics for the epoch.
        epoch_metrics = metrics_accumulator.compute()
        all_validation_metrics.append(epoch_metrics)

        # Prints epoch summary.
        print(f"Epoch {epoch:02d}")
        print(f"  Loss: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
        print(f"  MAE: {epoch_metrics['mae']:.6f}, RMSE: {epoch_metrics['rmse']:.6f}")
        print(f"  R²: {epoch_metrics['r2']:.4f}, Spearman: {epoch_metrics['spearman_corr']:.4f}")
        print(f"  SMAPE: {epoch_metrics['smape']:.2f}%, Max Error: {epoch_metrics['max_error']:.6f}")
        print(f"  Current LR: {current_lr:.6f}")

        # === Model Saving Logic ===
        is_best_model = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0 # Resets patience counter on improvement.
            model.eval() # Sets model to evaluation mode before saving.
            try:
                torch.save(model.state_dict(), BEST_MODEL_PATH) # Saves only the model's parameters.
                print(f"   Saved new best model to {BEST_MODEL_PATH}")
                is_best_model = True
            except Exception as e:
                print(f"   Failed to save model: {e}")
        else:
            patience_counter += 1 # Increments patience counter.
            print(f"   No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE or '∞'}")
            if EARLY_STOP_PATIENCE and patience_counter >= EARLY_STOP_PATIENCE:
                print(" Early stopping triggered. Training halted.")
                break # Exits training loop if early stopping condition met.

        # === Save Per-Epoch Summary JSON ===
        # Converts NumPy floats to standard Python floats for JSON serialization.
        summary_data_json_safe = {k: float(v) for k, v in epoch_metrics.items() if isinstance(v, (np.float32, np.float64))}
        
        with open(os.path.join(LOG_SUBDIRS["summary"], f"epoch_{epoch:02d}.json"), "w") as f:
            json.dump({
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "metrics": summary_data_json_safe,
                "lr": float(current_lr),
                "best_so_far": is_best_model,
                "best_epoch": best_epoch
            }, f, indent=2)

        # === Save Plots: Predicted vs True Scatter and Residual Histogram ===
        true_values = epoch_metrics['y_true']
        predicted_values = epoch_metrics['y_pred']
        
        # Scatter plot of true vs. predicted values.
        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predicted_values, alpha=0.5, s=10)
        plt.xlabel("True Delta Trace (Scaled)")
        plt.ylabel("Predicted Delta Trace (Scaled)")
        plt.title(f"Epoch {epoch:02d}: Predicted vs True")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_SUBDIRS["predictions"], f"epoch_{epoch:02d}_scatter.pdf"))
        plt.close()

        # Histogram of residuals (prediction errors).
        residuals = true_values - predicted_values
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
        plt.xlabel("Prediction Error (Residuals)")
        plt.ylabel("Frequency")
        plt.title(f"Epoch {epoch:02d}: Residual Histogram")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_SUBDIRS["residuals"], f"epoch_{epoch:02d}_residuals.pdf"))
        plt.close()

        # Stores epoch-level data for final curve plotting.
        epochs_history.append(epoch)
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        lrs_history.append(current_lr)

    # === Final Model Saving ===
    model.eval() # Sets model to evaluation mode for final saving.
    # Saves the last epoch's model state dictionary.
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "model_last.pt"))
    # Saves the complete model (architecture + state_dict).
    torch.save(model, os.path.join(MODEL_SAVE_DIR, "model_last_full.pt"))
    
    # If a best model was found during training, load and save its full version.
    if best_val_loss < float("inf"): # Checks if training actually occurred and a best model was recorded
        best_model_instance = TripleBranchNet(dropout=0.1).to(DEVICE)
        best_model_instance.load_state_dict(torch.load(BEST_MODEL_PATH))
        best_model_instance.eval()
        torch.save(best_model_instance, os.path.join(MODEL_SAVE_DIR, "model_best_full.pt"))
        print(f"Saved full best model from epoch {best_epoch}.")
    else:
        print("No best model saved as validation loss was not improved or no training occurred.")
    
    print("All models saved.")

    # === Save Training Logs and Final Plots ===
    # Prepares metrics for DataFrame, excluding large raw data arrays.
    metrics_for_dataframe = []
    for m in all_validation_metrics:
        m_copy = m.copy()
        m_copy.pop('y_true', None) # Remove raw data before creating DataFrame
        m_copy.pop('y_pred', None)
        metrics_for_dataframe.append(m_copy)
    
    # Creates a Pandas DataFrame from collected training metrics.
    df_training_log = pd.DataFrame({
        "Epoch": epochs_history,
        "Train Loss": train_losses_history,
        "Validation Loss": val_losses_history,
        "MAE": [m['mae'] for m in metrics_for_dataframe],
        "RMSE": [m['rmse'] for m in metrics_for_dataframe],
        "R2": [m['r2'] for m in metrics_for_dataframe],
        "SMAPE": [m['smape'] for m in metrics_for_dataframe],
        "Spearman Corr": [m['spearman_corr'] for m in metrics_for_dataframe],
        "Max Error": [m['max_error'] for m in metrics_for_dataframe],
        "Learning Rate": lrs_history
    })
    # Saves the training log to a CSV file.
    df_training_log.to_csv(os.path.join(LOG_SUBDIRS["curves"], "training_log.csv"), index=False)

    # Plots comprehensive learning curves.
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves (Train vs Validation).
    ax1.plot(epochs_history, train_losses_history, label="Train Loss")
    ax1.plot(epochs_history, val_losses_history, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.grid(True)
    ax1.legend()
    
    # MAE curve.
    ax2.plot(epochs_history, [m['mae'] for m in metrics_for_dataframe], label="MAE", color='green')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.set_title("Mean Absolute Error")
    ax2.grid(True)
    
    # Correlation curves (Spearman and Pearson).
    ax3.plot(epochs_history, [m['spearman_corr'] for m in metrics_for_dataframe], label="Spearman Correlation", color='red')
    ax3.plot(epochs_history, [m['pearson_corr'] for m in metrics_for_dataframe], label="Pearson Correlation", color='blue')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Correlation Coefficient")
    ax3.set_title("Correlation Metrics")
    ax3.grid(True)
    ax3.legend()
    
    # Learning Rate schedule curve.
    ax4.plot(epochs_history, lrs_history, label="Learning Rate", color='orange')
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Learning Rate")
    ax4.set_title("Learning Rate Schedule")
    ax4.grid(True)
    
    plt.tight_layout() # Adjusts plot parameters for a tight layout.
    plt.savefig(os.path.join(LOG_SUBDIRS["curves"], "comprehensive_curves.pdf"))
    plt.close() # Closes the plot to free memory.

    print("Logs and plots saved successfully.")
    print(f"\n🎉 Training complete! Best model achieved at epoch {best_epoch} with validation loss: {best_val_loss:.6f}")
    # Prints final metrics from the last epoch.
    if metrics_for_dataframe:
        final_metrics = metrics_for_dataframe[-1]
        print(f"Final R²: {final_metrics['r2']:.4f}")
        print(f"Final Spearman: {final_metrics['spearman_corr']:.4f}")
        print(f"Final MAE: {final_metrics['mae']:.6f}")
        print(f"Final SMAPE: {final_metrics['smape']:.2f}%")
    else:
        print("No final metrics to display as no training data was processed.")


# === Script Execution Point ===
if __name__ == "__main__":
    multiprocessing.freeze_support() # Required for multiprocessing on Windows.
    train_model() # Initiates the model training process.
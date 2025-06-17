import os
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration Paths ===
# Defines the base directory of the processed dataset.
# This path points to the specific timestamped output folder from the cleaning step.
# Example: "processed_datasets/dataset_combined_mixed_20250429_193157"
BASE_DATASET_DIR = os.path.join("processed_datasets", "dataset_combined_mixed_YYYYMMDD_HHMMSS") # <--- MANUAL UPDATE REQUIRED

# Defines the timestamp of the specific training run whose logs are to be plotted.
# This timestamp corresponds to the `output/YYYYMMDD_HHMMSS` folder created during training.
TRAINING_RUN_TIMESTAMP = "YYYYMMDD_HHMMSS" # <--- MANUAL UPDATE REQUIRED

# Constructs the full path to the training log CSV file.
LOG_FILE_PATH = os.path.join(BASE_DATASET_DIR, "output", TRAINING_RUN_TIMESTAMP, "logs", "curves", "training_log.csv")
# Sets the output directory for the generated plots to the same location as the log file.
OUTPUT_PLOT_DIR = os.path.dirname(LOG_FILE_PATH)

# === Load Training Log ===
print(f"Loading training log from: {LOG_FILE_PATH}")
# Reads the CSV file into a Pandas DataFrame.
try:
    df_training_log = pd.read_csv(LOG_FILE_PATH)
    print(" Training log loaded successfully.")
except FileNotFoundError:
    print(f" Error: Training log file not found at {LOG_FILE_PATH}. Please check the path.")
    exit()
except Exception as e:
    print(f" Error loading training log: {e}")
    exit()

# === Define Metrics for Plotting ===
# A list of dictionaries, each defining a metric to be plotted.
# Includes the column name, plot color, and Y-axis label.
METRICS_TO_PLOT = [
    {"name": "Train Loss", "color": "blue", "ylabel": "Loss"},
    {"name": "Validation Loss", "color": "orange", "ylabel": "Loss"},
    {"name": "MAE", "color": "green", "ylabel": "Mean Absolute Error"},
    {"name": "RMSE", "color": "red", "ylabel": "Root Mean Square Error"},
    {"name": "R2", "color": "purple", "ylabel": "R² Score"},
    {"name": "SMAPE", "color": "brown", "ylabel": "Symmetric Mean Absolute Percentage Error (%)"},
    {"name": "Spearman Corr", "color": "teal", "ylabel": "Spearman Correlation"},
    {"name": "Max Error", "color": "magenta", "ylabel": "Maximum Absolute Error"},
    {"name": "Learning Rate", "color": "gray", "ylabel": "Learning Rate"}
]

# === Plot Generation Function ===
def create_metric_plot(dataframe: pd.DataFrame, metric_name: str, color: str, y_label: str, output_directory: str) -> str:
    """
    Generates and saves a single plot for a specified metric over epochs.

    Args:
        dataframe: The Pandas DataFrame containing the training log data.
        metric_name: The name of the metric column to plot.
        color: The color for the plot line.
        y_label: The label for the Y-axis.
        output_directory: The directory where the plot will be saved.

    Returns:
        The file path of the saved plot.
    """
    plt.figure(figsize=(8, 5)) # Sets the figure size for the plot.
    plt.plot(dataframe["Epoch"], dataframe[metric_name], label=metric_name, color=color)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(f"{metric_name} per Epoch")
    plt.grid(True) # Adds a grid to the plot for better readability.
    plt.tight_layout() # Adjusts plot parameters for a tight layout.
    
    # Creates a sanitized filename by replacing spaces with underscores and converting to lowercase.
    filename = f"{metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('²', '2')}_curve.pdf"
    plot_path = os.path.join(output_directory, filename)
    plt.savefig(plot_path) # Saves the plot to the specified path.
    plt.close() # Closes the plot figure to free up memory.
    print(f" Saved plot: {os.path.basename(plot_path)}")
    return plot_path

# === Generate All Plots ===
print("\nGenerating and saving individual metric plots...")
generated_plot_paths = []
for metric in METRICS_TO_PLOT:
    plot_path = create_metric_plot(df_training_log, metric["name"], metric["color"], metric["ylabel"], OUTPUT_PLOT_DIR)
    generated_plot_paths.append(plot_path)

# === Summary ===
print(f"\n=== Plot Generation Summary ===")
print(f"Total plots created: {len(generated_plot_paths)}")
print(f"All plots saved to: {OUTPUT_PLOT_DIR}")
print(" Plot generation complete!")
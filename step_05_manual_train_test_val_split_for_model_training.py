import os
import json
from sklearn.model_selection import train_test_split

# === Configuration ===
# Defines the base directory of the cleaned dataset.
# This path should point to the specific timestamped output folder from the cleaning step.
# Example: "processed_datasets/dataset_combined_mixed_20250429_193157"
BASE_DIR = os.path.join("processed_datasets", "dataset_combined_mixed_YYYYMMDD_HHMMSS") # <--- MANUAL UPDATE REQUIRED

# Defines the path to the input JSONL file, which is the output from the cleaning script.
INPUT_JSONL_PATH = os.path.join(BASE_DIR, "dataset_cleaned.jsonl")

# Defines the output file paths for the generated train, validation, and test datasets.
OUTPUT_FILES = {
    "train": os.path.join(BASE_DIR, "dataset_train.jsonl"),
    "val": os.path.join(BASE_DIR, "dataset_val.jsonl"),
    "test": os.path.join(BASE_DIR, "dataset_test.jsonl"),
}

# Sets a random seed for reproducibility of the data splitting process.
RANDOM_SEED = 42

# === Load Dataset Lines ===
print("Loading dataset lines for splitting...")
# Reads all non-empty lines from the input JSONL file.
with open(INPUT_JSONL_PATH, "r") as f:
    LINES = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(LINES)} total lines from {INPUT_JSONL_PATH}.")

# === Split Dataset ===
print("Performing train/validation/test split (80/10/10 ratio)...")

# First split: Divides the data into an 80% training set and a 20% temporary set.
TRAIN_LINES, TEMP_LINES = train_test_split(
    LINES, test_size=0.2, random_state=RANDOM_SEED
)

# Second split: Divides the 20% temporary set equally into 10% validation and 10% test sets.
VAL_LINES, TEST_LINES = train_test_split(
    TEMP_LINES, test_size=0.5, random_state=RANDOM_SEED
)

# Consolidates the split data into a dictionary for easy iteration.
SPLITS = {"train": TRAIN_LINES, "val": VAL_LINES, "test": TEST_LINES}

# === Save Split Datasets ===
print("Saving split datasets...")
for name, data in SPLITS.items():
    # Writes each split to its designated output file.
    with open(OUTPUT_FILES[name], "w") as f_out:
        for line in data:
            item = json.loads(line)
            # Normalizes the entry: flattens if nested under a "jsonl" key, otherwise uses as is.
            normalized_entry = item.get("jsonl", item)
            f_out.write(json.dumps(normalized_entry) + "\n")

    print(f"Saved {len(data)} lines to {OUTPUT_FILES[name]}")

print("\nDataset splitting complete!")
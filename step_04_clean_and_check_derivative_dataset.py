import os
import json
import numpy as np

# === Configuration ===
# Defines the base directory of the processed dataset to be cleaned.
# This path points to the specific timestamped output folder from the derivative dataset creation step.
# Example: "processed_datasets/dataset_combined_mixed_20250429_193157"
BASE_DIR = os.path.join("processed_datasets", "dataset_combined_mixed_YYYYMMDD_HHMMSS") # <--- MANUAL UPDATE REQUIRED

# Defines the path to the input JSONL file within the base directory.
JSONL_PATH = os.path.join(BASE_DIR, "dataset.jsonl")
# Defines the directory containing the density matrix (.npz) files within the base directory.
DM_DIR = os.path.join(BASE_DIR, "dm")
# Defines the output path for the new, cleaned JSONL file.
CLEANED_JSONL_PATH = os.path.join(BASE_DIR, "dataset_cleaned.jsonl")

# Controls whether corrupted .npz files are permanently deleted or merely logged.
DELETE_BAD_NPZ = True # Set to False to only log corrupted files without deleting

# === Step 1: Clean and deduplicate dataset.jsonl entries ===
print("\nStep 1: Cleaning dataset.jsonl and removing duplicates...")
# Stores unique keys to identify and filter out duplicate entries in the JSONL file.
seen_entries = set()
# Accumulates valid and deduplicated JSONL entries.
cleaned_jsonl_entries = []
# Tracks the filenames of .npz files referenced by the valid JSONL entries.
referenced_dm_files = set()
# Counts the number of lines in the JSONL file that could not be parsed.
skipped_bad_lines = 0

with open(JSONL_PATH, "r") as f:
    for line in f:
        try:
            item = json.loads(line)
            # Extracts the core data entry, accommodating for nested "jsonl" keys.
            entry = item["jsonl"] if "jsonl" in item else item
            # Extracts the filename of the density matrix referenced within the entry.
            dm_filename = os.path.basename(entry["input"]["density_matrix_path"])
            # Creates a unique, sorted key from input and target data to detect duplicates reliably.
            unique_key = json.dumps(entry["input"], sort_keys=True) + json.dumps(entry["target"], sort_keys=True)

            if unique_key in seen_entries:
                continue # Skips the current line if it is a duplicate entry.
            
            seen_entries.add(unique_key)
            cleaned_jsonl_entries.append({"jsonl": entry}) # Adds the cleaned entry to the list.
            referenced_dm_files.add(dm_filename) # Records the .npz file as referenced.

        except Exception as e:
            print(f"Skipping malformed JSON line: {str(e)}")
            skipped_bad_lines += 1

print(f"Cleaned JSONL entries: {len(cleaned_jsonl_entries)} (skipped {skipped_bad_lines} malformed lines)")

# === Step 2: Scan .npz files for corruption and handle them ===
print("\nStep 2: Scanning .npz files for integrity...")
# Stores the filenames of .npz files that are successfully loaded and determined to be valid.
valid_dm_files = set()
# Stores the filenames of .npz files that are found to be corrupt.
corrupt_dm_files = set()

for filename in os.listdir(DM_DIR):
    if not filename.endswith(".npz"):
        continue # Proceeds only with files having a .npz extension.
    
    file_path = os.path.join(DM_DIR, filename)
    try:
        with np.load(file_path) as data:
            _ = data["dm"] # Attempts to load and access the 'dm' key to verify file integrity.
        valid_dm_files.add(filename) # Marks the file as valid if loading is successful.
    except Exception as e:
        print(f"Corrupt .npz file detected: {filename} ({str(e)})")
        corrupt_dm_files.add(filename) # Marks the file as corrupt.
        if DELETE_BAD_NPZ:
            try:
                os.remove(file_path) # Deletes the corrupt file if the `DELETE_BAD_NPZ` flag is True.
                print(f"Deleted corrupt file: {filename}")
            except PermissionError as pe:
                print(f"Permission denied. Could not delete {filename}: {pe}")

print(f"Valid .npz files found: {len(valid_dm_files)}")
print(f"Corrupt .npz files: {len(corrupt_dm_files)}")

# === Step 3: Filter JSONL entries to only include those with valid and existing .npz files ===
print("\nStep 3: Filtering JSONL entries to match valid .npz files...")
# Stores the final set of JSONL entries that are valid and whose referenced .npz files exist and are not corrupt.
final_filtered_entries = []
for entry in cleaned_jsonl_entries:
    dm_filename = os.path.basename(entry["jsonl"]["input"]["density_matrix_path"])
    if dm_filename in valid_dm_files:
        final_filtered_entries.append(entry) # Adds the entry if its corresponding .npz file is valid.

print(f"Final valid entries after filtering: {len(final_filtered_entries)}")

# === Step 4: Save the cleaned and filtered dataset to a new JSONL file ===
with open(CLEANED_JSONL_PATH, "w") as f:
    for entry in final_filtered_entries:
        f.write(json.dumps(entry) + "\n")

print(f"\nSaved cleaned dataset to: {CLEANED_JSONL_PATH}")

# === Step 5: Identify and remove orphaned .npz files (not referenced by the final JSONL) ===
print("\nStep 5: Removing orphaned .npz files...")
# Creates a set of .npz filenames that are explicitly referenced by entries in the `final_filtered_entries` list.
used_dm_filenames = {
    os.path.basename(e["jsonl"]["input"]["density_matrix_path"]) for e in final_filtered_entries
}

# Identifies .npz files that are valid but are not found in the `used_dm_filenames` set, indicating they are orphans.
orphaned_dm_files = valid_dm_files - used_dm_filenames
for orphan_filename in orphaned_dm_files:
    file_path = os.path.join(DM_DIR, orphan_filename)
    try:
        os.remove(file_path) # Deletes the orphaned file.
        print(f"Removed orphan .npz file: {orphan_filename}")
    except PermissionError as pe:
        print(f"Permission denied. Could not delete orphan {orphan_filename}: {pe}")

print(f"\nOrphan .npz files removed: {len(orphaned_dm_files)}")
print(f"Dataset cleaning complete!")
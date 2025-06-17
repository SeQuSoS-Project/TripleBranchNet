import os
import shutil
import time
import random

# --- Configuration ---
# Define the root directory for the combined datasets.
# This will be created relative to where the script is run.
OUTPUT_ROOT = "combined_datasets" # Using a relative path for portability

# Maximum number of files to combine in the output dataset.
# This limit applies to the total number of files (test + train).
LIMIT = 1000 

# List of source directories for the dataset files.
# Each tuple contains a tag (for logging/naming) and the path to the source folder.
# IMPORTANT: These paths are relative to the 'mnisq_cache' directory created by the
# download script step_01_load_mnisq_dataset.py. Ensure 'mnisq_cache' is in the same parent directory as this script,
# or adjust 'DOWNLOAD_CACHE_DIR' accordingly if it's located elsewhere.
# Currently configured for 'mnist_784' dataset. Modify paths to include
# 'Fashion-MNIST' or 'Kuzushiji-MNIST' if different datasets are desired.
DOWNLOAD_CACHE_DIR = "mnisq_cache" # Base directory from the step_01_load_mnisq_dataset.py script
ALL_SOURCES = [
    ("f80_test", os.path.join(DOWNLOAD_CACHE_DIR, "base_test_mnist_784_f80")),
    ("f90_test", os.path.join(DOWNLOAD_CACHE_DIR, "base_test_mnist_784_f90")),
    ("f95_test", os.path.join(DOWNLOAD_CACHE_DIR, "base_test_mnist_784_f95")),
    ("f80_train", os.path.join(DOWNLOAD_CACHE_DIR, "base_train_orig_mnist_784_f80")),
    ("f90_train", os.path.join(DOWNLOAD_CACHE_DIR, "base_train_orig_mnist_784_f90")),
    ("f95_train", os.path.join(DOWNLOAD_CACHE_DIR, "base_train_orig_mnist_784_f95")),
]

# --- Utility Functions ---

def load_existing_used_files(output_root_path: str) -> set:
    """
    Loads file IDs and their source folders from existing log files within
    the output root, preventing duplicate usage across different combined datasets.

    Args:
        output_root_path: The root directory where combined datasets are stored.

    Returns:
        A set of tuples, where each tuple is (file_id, source_folder_path).
    """
    used_files = set()
    for root, _, files in os.walk(output_root_path):
        if "log_used_files.txt" in files:
            log_path = os.path.join(root, "log_used_files.txt")
            with open(log_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        file_id = parts[0].strip()
                        source_folder = parts[1].strip()
                        used_files.add((file_id, source_folder))
    return used_files

def get_file_ids(folder_path: str) -> list:
    """
    Retrieves a sorted list of file IDs (filenames) from the 'qasm' subdirectory
    of a given dataset folder.

    Args:
        folder_path: The base path to the dataset folder (e.g., 'mnisq_cache/base_test_mnist_784_f80').

    Returns:
        A list of filenames found in the 'qasm' subdirectory.
    """
    qasm_path = os.path.join(folder_path, "qasm")
    if not os.path.exists(qasm_path):
        print(f"Warning: QASM directory not found at {qasm_path}. Skipping source.")
        return []
    return sorted(os.listdir(qasm_path))

def make_output_dirs(base_name: str) -> str:
    """
    Creates the necessary directory structure for a new combined dataset.
    This includes 'qasm', 'state', 'label', and 'fidelity' subdirectories.

    Args:
        base_name: The desired name for the new combined dataset directory.

    Returns:
        The full path to the newly created base directory for the combined dataset.
    """
    path = os.path.join(OUTPUT_ROOT, base_name)
    for sub in ["qasm", "state", "label", "fidelity"]:
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    return path

def copy_sample(src_base: str, dst_base: str, file_id: str, tag: str):
    """
    Copies all components (qasm, state, label, fidelity) of a single sample
    from a source folder to a destination folder, renaming files with a tag
    while preserving the original file extension.

    Args:
        src_base: The base source directory of the sample.
        dst_base: The base destination directory for the combined dataset.
        file_id: The unique identifier (original filename) of the sample.
        tag: A tag to append to the new filenames for identification.
    """
    # Split the original file_id into its base name and extension
    base_name, ext = os.path.splitext(file_id)

    for sub in ["qasm", "state", "label", "fidelity"]:
        src_file = os.path.join(src_base, sub, file_id)
        # Construct the new filename: original_basename_tag.original_extension
        new_name = f"{base_name}_{tag}{ext}"
        dst_file = os.path.join(dst_base, sub, new_name)
        shutil.copy(src_file, dst_file)

# --- Main Logic ---

def randomly_sample_and_copy(sources: list, output_tag: str, limit: int):
    """
    Randomly samples files from various source directories, copies them to
    a new combined dataset directory, and logs the used files to avoid duplication.

    Args:
        sources: A list of (tag, path) tuples for all source directories.
        output_tag: A tag to be included in the name of the output directory
                    and appended to copied filenames.
        limit: The maximum number of files to copy into the combined dataset.
    """
    # Create a timestamped directory for the new combined dataset.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = make_output_dirs(f"combined_{output_tag}_{timestamp}")
    log_path = os.path.join(output_dir, "log_used_files.txt")

    # Load previously used files to prevent copying the same file multiple times
    # across different runs or combined datasets within the OUTPUT_ROOT.
    used_global = load_existing_used_files(OUTPUT_ROOT)
    counter = 0

    # Prepare streams: For each source, get all file IDs and shuffle them.
    streams = []
    for tag, path in sources:
        files = get_file_ids(path)
        if files: # Only add sources that actually contain files
            random.shuffle(files)
            streams.append({"tag": tag, "path": path, "files": files})
        else:
            print(f"Skipping empty or invalid source: {path}")

    # Open log file to record which files are copied in this run.
    with open(log_path, "w") as log:
        while counter < limit:
            # Filter for streams that still have files available.
            available_streams = [s for s in streams if s["files"]]
            if not available_streams:
                print("[!] All available sources exhausted before reaching the limit.")
                break # Exit if no more files can be sampled.

            # Randomly select a source stream and pop a file ID.
            selected_stream = random.choice(available_streams)
            file_id = selected_stream["files"].pop()
            source_folder = selected_stream["path"]
            entry = (file_id, source_folder)

            # Check if this file has already been used in any previous combined dataset.
            if entry in used_global:
                continue # Skip if already used.

            # Copy the sample and update the tracking sets.
            copy_sample(source_folder, output_dir, file_id, f"{output_tag}_{selected_stream['tag']}")
            used_global.add(entry)
            log.write(f"{file_id},{source_folder},{output_tag}_{selected_stream['tag']}\n")
            counter += 1

    print(f"\nSuccessfully created: {output_dir}")
    print(f"Total files combined: {counter} out of a limit of {limit}.")
    print(f"Log of used files written to: {log_path}")

# --- Execution ---
if __name__ == "__main__":
    print("Starting dataset combination process...")
    # Call the main function to create the combined dataset.
    # The 'mixed' tag will be part of the output folder name and file names.
    randomly_sample_and_copy(ALL_SOURCES, "mixed", LIMIT)
    print("\nDataset combination process complete.")
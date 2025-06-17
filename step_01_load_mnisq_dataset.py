import os
import requests
import zipfile
import shutil
import logging

# --- Configuration ---
# Define the directory for all downloaded and extracted files.
DOWNLOAD_BASE_DIR = "mnisq_cache"
ARCHIVE_DIR = os.path.join(DOWNLOAD_BASE_DIR, "archives")
LOG_FILE = os.path.join(DOWNLOAD_BASE_DIR, "download_log.log")

# Create necessary directories if they don't already exist.
os.makedirs(DOWNLOAD_BASE_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Configure logging to record download and extraction activities.
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Base URL for the datasets.
BASE_URL = "https://qulacs-quantum-datasets.s3.us-west-1.amazonaws.com/"

# Dataset parameters: types, splits, and fidelities.
DATASET_TYPES = ["mnist_784", "Fashion-MNIST", "Kuzushiji-MNIST"]
DATASET_SPLITS = ["base_test", "base_train_orig"]
DATASET_FIDELITIES = ["f80", "f90", "f95"]

# --- Main Download and Extraction Logic ---
print("Starting dataset download and extraction process...")

for dtype in DATASET_TYPES:
    for split in DATASET_SPLITS:
        for fidelity in DATASET_FIDELITIES:
            zip_filename = f"{split}_{dtype}_{fidelity}.zip"
            download_url = f"{BASE_URL}{zip_filename}"
            
            # Define paths for the downloaded ZIP and its extracted folder.
            local_zip_path = os.path.join(DOWNLOAD_BASE_DIR, zip_filename)
            extracted_folder_path = os.path.join(DOWNLOAD_BASE_DIR, zip_filename.replace(".zip", ""))
            archive_zip_path = os.path.join(ARCHIVE_DIR, zip_filename)

            # Check if both the ZIP file and the extracted folder already exist.
            if os.path.exists(archive_zip_path) and os.path.exists(extracted_folder_path):
                print(f"Skipping {zip_filename} — already downloaded and extracted.")
                logging.info(f"Skipped: {zip_filename} (already exists)")
                continue

            try:
                # Download the ZIP file if it's not already in the archive.
                if not os.path.exists(archive_zip_path):
                    print(f"Downloading {zip_filename} ...")
                    response = requests.get(download_url)
                    response.raise_for_status()  # Raise an exception for bad status codes
                    with open(local_zip_path, "wb") as f:
                        f.write(response.content)
                    logging.info(f"Downloaded: {zip_filename}")
                else:
                    print(f"Found ZIP in archive: {zip_filename} — skipping download.")

                # Extract the contents of the ZIP file if the folder doesn't exist.
                if not os.path.exists(extracted_folder_path):
                    print(f"Extracting {zip_filename} ...")
                    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
                        zip_ref.extractall(DOWNLOAD_BASE_DIR)
                    logging.info(f"Extracted: {zip_filename}")
                else:
                    print(f"Already extracted: {zip_filename} — skipping unzip.")

                # Move the downloaded ZIP file to the archive directory if it's still in the base directory.
                if os.path.exists(local_zip_path) and not os.path.exists(archive_zip_path):
                    shutil.move(local_zip_path, archive_zip_path)
                    print(f"Moved {zip_filename} to archives/.")
                    logging.info(f"Moved to archive: {zip_filename}")

                print(f"Successfully processed: {zip_filename}")

            except requests.exceptions.RequestException as e:
                # Handle network-related errors during download.
                logging.error(f"Network error downloading {zip_filename}: {e}")
                print(f"Error downloading {zip_filename}: {e}")
            except zipfile.BadZipFile as e:
                # Handle corrupted ZIP file errors.
                logging.error(f"Bad ZIP file {zip_filename}: {e}")
                print(f"Error extracting {zip_filename}: {e}")
            except Exception as e:
                # Catch any other unexpected errors.
                logging.error(f"An unexpected error occurred with {zip_filename}: {e}")
                print(f"An unexpected error occurred with {zip_filename}: {e}")

print("\nDataset processing complete.")
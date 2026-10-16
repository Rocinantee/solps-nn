import os
import sys
import hashlib
import requests
import json

from .main import species_dict

# The path to the package directory
package_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(package_dir,'model_data')

# Load dataset information from config file
def load_config(config_path=None):
    """Load the stored model description."""
    if config_path == None:
        config_path = os.path.join(package_dir,'config.json')
    with open(config_path, "r") as f:
        return json.load(f)

def verify_file(expected_hash,file_path):
    """Verify whether file exists and is correct."""
    if os.path.exists(file_path):
        if os.path.isfile(file_path):
            actual_hash = calculate_file_hash(file_path)
            if actual_hash == expected_hash:
                print(f"{file_path} is up-to-date.")
                return True
            else:
                print(f"{file_path} is corrupt (hash mismatch). Re-downloading...")
                return False
        else:
            raise FileExistsError(f"{file_path} exists but is not a regular file.")
    else:
        print(f"{file_path} is missing. Downloading...")
        return False

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def download_file(url, file_path):
    """Download a file from a URL and save it to the given path."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_model(item='te',species=None,config_path=None):
    """Verify file integrity and download missing/corrupt files."""
    model_files = load_config(config_path)

    if species in species_dict:
        model_name = item + str(species_dict[species])
    else:
        model_name = item
    
    if model_name in model_files:
        print(f"Requested model: {model_name}")
        model_names = ['root',model_name]
    elif model_name == 'all':
        print("All models requested.")
        model_names = [name for name in model_files]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    for model_name in model_names:
        for file_info in model_files[model_name]:
            file_path = os.path.join(data_folder, file_info["filename"])
            expected_hash = file_info["hash"]
            if not verify_file(expected_hash,file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                download_file(file_info["url"], file_path)
                if calculate_file_hash(file_path) != expected_hash:
                    os.remove(file_path)
                    raise IOError(f"Downloaded {file_info['filename']} is corrupt (hash mismatch). Please retry the download.")
                print(f"Downloaded {file_info['filename']}.")
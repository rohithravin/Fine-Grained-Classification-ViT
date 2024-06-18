import os
import requests
import zipfile
import tarfile
import io
import torch

import config

def compute_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print ("MPS device found.")
    else:
        device = 'cpu'
        print ("MPS device not found.")
    return device

def download_and_extract_tar(url = config.STANFORD_DOG_DATASET_URL, extract_to='.', rename_folder_to='stanford-dog-dataset'):
    """
    Download a tar file from a URL and extract it to the specified directory. Optionally, rename the extracted folder.

    Args:
    url (str): The URL of the tar file to download.
    extract_to (str): The directory to extract the files to. Defaults to current directory.
    rename_folder_to (str or None): The new name for the extracted folder. If None, the folder won't be renamed.

    Raises:
    Exception: If there's an issue with downloading or extracting the file.
    """

    new_folder_name = os.path.join(extract_to, rename_folder_to)
    if os.path.exists(new_folder_name):
        print(f"The folder {new_folder_name} already exists.") 
        return

    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Check if the download was successful

        # Extract the tar file
        with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:*') as tar_ref:
            tar_ref.extractall(extract_to)
            print(f"Files extracted to {extract_to}")

            # Determine the name of the extracted folder
            extracted_folders = [member.name.split('/')[0] for member in tar_ref.getmembers() if member.isdir()]
            if extracted_folders:
                extracted_folder_name = os.path.join(extract_to, extracted_folders[0])
                if rename_folder_to:
                    if os.path.exists(new_folder_name):
                        print(f"The folder {new_folder_name} already exists. No renaming will be done.")
                    else:
                        os.rename(extracted_folder_name, new_folder_name)
                        print(f"Renamed {extracted_folder_name} to {new_folder_name}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except tarfile.TarError as e:
        print(f"Error extracting the tar file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Iterate over subfolders
    for subfolder in os.listdir(new_folder_name):
        subfolder_path = os.path.join(new_folder_name, subfolder)
        if os.path.isdir(subfolder_path):
            # Get the part after the first hyphen
            new_subfolder_name = subfolder.split('-', 1)[-1]
            # Rename the subfolder
            os.rename(subfolder_path, os.path.join(new_folder_name, new_subfolder_name))

    
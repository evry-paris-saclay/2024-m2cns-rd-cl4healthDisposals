import os
import requests
import zipfile
import tarfile


def download_and_extract(url, extract_to="data"):
    os.makedirs(extract_to, exist_ok=True)
    filename = url.split("/")[-1]
    file_path = os.path.join(extract_to, filename)

    # Download File
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded {filename} to {file_path}.")
    else:
        print(f"Failed to download {url}. HTTP Status Code: {response.status_code}")
        return

    # Create target folder for extracted files
    target_folder = os.path.join(extract_to, "Medical Waste 4.0")
    os.makedirs(target_folder, exist_ok=True)

    # Unzip the file
    print(f"Extracting {filename}...")
    if filename.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(target_folder)
            print(f"Extracted {filename} to {target_folder}.")
    elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(target_folder)
            print(f"Extracted {filename} to {target_folder}.")
    elif filename.endswith(".tar"):
        with tarfile.open(file_path, "r:") as tar_ref:
            tar_ref.extractall(target_folder)
            print(f"Extracted {filename} to {target_folder}.")
    else:
        print(f"Unsupported file format for {filename}. Skipping extraction.")

    # Delete the compressed file (optional)
    os.remove(file_path)
    print(f"Removed the downloaded file {file_path}.")


dataset_url = "https://zenodo.org/record/7643417/files/dataset.zip"
download_and_extract(dataset_url, extract_to="data")

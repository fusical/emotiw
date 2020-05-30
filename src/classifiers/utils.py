import os
import shutil
import zipfile
from pathlib import Path

def unzip_folder(folder, tmp_file_name):
    ""
    if folder.endswith(".zip"):
        # Unzips files to a temp directory
        tmp_output_folder = tmp_file_name
        if os.path.exists(tmp_output_folder) and os.path.isdir(tmp_output_folder):
            print("Removing existing dir...")
            shutil.rmtree(tmp_output_folder)

        print(f"Unzipping files to temp dir {tmp_output_folder}...")
        Path(f"{tmp_output_folder}").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(folder, 'r') as zip_ref:
            zip_ref.extractall(tmp_output_folder)
        print("Finished unzipping files")
    else:
        tmp_output_folder = folder
        print("Skipping unzipping files as input is a folder")
    return tmp_output_folder

def get_num_samples(folder):
    path, dirs, files = next(os.walk(folder))
    return len(files)

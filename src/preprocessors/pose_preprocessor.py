import zipfile
import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from os.path import isfile, join
import pickle
import subprocess
import os


class PosePreprocessor:
    """
    Extract the poses from the video frames.
    """

    def __init__(self, video_frame_folder, output_folder, output_file=None, is_test=False):
        """
        @param video_frame_folder    The folder where the list of videos frames are stored. If
                                     this ends with .zip, this should be a single zip
                                     file containing the video frames. Paths can either by a local
                                     folder or a GDrive mounted path.
        @param output_folder         The local output path where the preprocessed files will be stored for
                                     further preprocessing can be done
        @param output_file           If not none, the output_folder will be zipped up and stored at this location
        @param is_test               If set to True, the `video_frame_folder` is assumed to have no categorical
                                     classification folder hierarchy
        """
        self.is_test = is_test
        self.video_frame_folder = video_frame_folder
        self.output_folder = output_folder
        self.output_file = output_file
        print(
            f"Pose Preprocessor created with is_test = {is_test}, video_frame_folder = {video_frame_folder} , output_folder = {output_folder}, output_file = {output_file}")

    def preprocess(self):
        tmp_output_folder = ""
        if self.video_frame_folder.endswith(".zip"):
            # Unzips files to a temp directory
            tmp_output_folder = self.output_folder.rstrip('/') + "_tmp"
            print(f"Unzipping files to temp dir {tmp_output_folder}...")
            Path(f"{tmp_output_folder}").mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.video_frame_folder, 'r') as zip_ref:
                zip_ref.extractall(tmp_output_folder)
            print("Finished unzipping files")
        else:
            tmp_output_folder = self.video_frame_folder
            print("Skipping unzipping files as input is a folder")

        # Create output folder for the keypoints
        Path(f"{self.output_folder}").mkdir(parents=True, exist_ok=True)

        if self.is_test:
            print(f"Starting test pose extraction for {tmp_output_folder}")
            p = subprocess.run(
                ["build/examples/openpose/openpose.bin", "--image_dir", "../" + tmp_output_folder, "--write_json",
                 "../" + self.output_folder, "--display", "0", "--render_pose", "0"], cwd="openpose",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            print(p.stdout)
            print(p.stderr)
        else:
            print(f"Starting pose extraction for {tmp_output_folder}")
            # Subfolders represent the different categories
            # (we will mimic this for the final output)
            subfolders = next(os.walk(tmp_output_folder))[1]
            for subfolder in subfolders:
                print(f"Starting pose extraction for {join(tmp_output_folder, subfolder)}")
                p = subprocess.run(
                    ["build/examples/openpose/openpose.bin", "--image_dir", "../" + join(tmp_output_folder, subfolder),
                     "--write_json", "../" + join(self.output_folder, subfolder), "--display", "0", "--render_pose",
                     "0"], cwd="openpose", stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                print(p.stdout)
                print(p.stderr)

        if self.output_file is not None:
            print(f"Starting to zip files to {self.output_file}")

            def zipdir(path, ziph):
                for root, dirs, files in os.walk(path):
                    folder = root[len(path):]
                    for file in files:
                        ziph.write(join(root, file), join(folder, file))

            zipf = zipfile.ZipFile(self.output_file, 'w', zipfile.ZIP_DEFLATED)
            zipdir(self.output_folder, zipf)
            zipf.close()
            print(f"Done zipping files to {self.output_file}")

        print("Done!")

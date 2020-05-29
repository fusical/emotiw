import zipfile
import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from os.path import isfile, join


class VideoPreprocessor:
    """
    Preprocesses raw videos into frames so that further downstream extraction and
    preprocessing can occur
    Example usage:
    video_preprocessor = VideoPreprocessor(
        video_folder="drive/My Drive/cs231n-project/datasets/emotiw/train-tiny.zip",
        label_file="drive/My Drive/cs231n-project/datasets/emotiw/Train_labels.txt",
        output_folder="train-tiny-local",
        output_file="drive/My Drive/cs231n-project/datasets/emotiw/train-tiny-local.zip"
    )
    video_preprocessor.preprocess()
    """

    def __init__(self, video_folder, output_folder, output_file=None, label_file=None, is_zip=True, height=320,
                 width=480, sample_every=10, max_workers=32):
        """
        @param video_folder   The folder where the list of videos are stored. If
                              `is_zip` is set to True, this should be a single zip
                              file containing the videos. Paths can either by a local
                              folder or a GDrive mounted path.
        @param label_file     The file containing the space-delimited video name to label mapping. If None,
                              we assume that we are in 'test' mode and that there are no categories
        @param output_folder  The local output path where the preprocessed files will be stored for
                              further preprocessing can be done
        @param output_file    If not none, the output_folder will be zipped up and stored at this location
        @param is_zip         If set to True, the `video_folder` will be unzipped prior to accessing
        @param height         Height of the extracted video frames
        @param width          Width of the extracted video frames
        @param sample_every   The frames to skip.
        @param max_workers    The number of workers to use to parallelize work.
        """
        self.is_zip = is_zip
        self.video_folder = video_folder
        self.label_file = label_file
        self.output_folder = output_folder
        self.output_file = output_file
        print(
            f"Video Preprocessor created with is_zip = {is_zip}, video_folder = {video_folder} , label_file = {label_file} , output_folder = {output_folder}, output_file = {output_file}")

        self.height = height
        self.width = width
        self.sample_every = sample_every
        self.max_workers = max_workers
        print(f"Frames will be created with height = {height} , width = {width} , sample_every = {sample_every}")

    def preprocess(self):
        if self.is_zip:
            # Unzips files to a temp directory
            tmp_output_folder = self.output_folder.rstrip('/') + "_tmp"
            print(f"Unzipping files to temp dir {tmp_output_folder}...")
            Path(f"{tmp_output_folder}").mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.video_folder, 'r') as zip_ref:
                zip_ref.extractall(tmp_output_folder)
            print("Finished unzipping files")
        else:
            tmp_output_folder = self.video_folder
            print("Skipping unzipping files as input is a folder")

        if self.label_file is not None:
            # Create the category subfolders in the output folder
            # Path Structure:
            #   output/
            #     1/
            #     2/
            video_to_label = {}
            unique_labels = set()
            with open(self.label_file, "r") as f:
                i = 0
                for line in f:
                    if i == 0:
                        i += 1
                        continue
                    line_arr = line.split(" ")
                    video_to_label[line_arr[0] + ".mp4"] = line_arr[1].strip()
                    if line_arr[1].strip() not in unique_labels:
                        unique_labels.add(line_arr[1].strip())
                        Path(f"{self.output_folder}/{line_arr[1].strip()}").mkdir(parents=True, exist_ok=True)
                    i += 1
        else:
            video_to_label = None
            Path(f"{self.output_folder}").mkdir(parents=True, exist_ok=True)

        # Process each video by extracting the frames in a multi-threaded fashion
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            videos = next(os.walk(tmp_output_folder))[2]
            videos = sorted(videos)

            print(f"Found {len(videos)} videos")
            video_num = 1

            for video_name in videos:
                future = executor.submit(self.process_video, tmp_output_folder, video_name, video_num, len(videos),
                                         video_to_label)
                futures.append(future)
                video_num += 1

        cv2.destroyAllWindows()

        print("***** Submitted all tasks *****")
        for future in futures:
            future.result()
        print("***** Completed *****")

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

    def process_video(self, tmp_output_folder, video_name, video_num, total_videos, video_to_label):
        """
        Processes a video by extracting the frames
        Writes out each frame, where each frame is an image resized to the the specified dimensions
        """
        vidcap = cv2.VideoCapture(join(tmp_output_folder, video_name))
        if video_to_label is not None:
            label = video_to_label[video_name]
            print(f"Processing video {video_num}/{total_videos} with name {video_name} and class {label} \n")
        else:
            label = None
            print(f"Processing video {video_num}/{total_videos} with name {video_name} (test-mode) \n")

        input_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))

        success, image = vidcap.read()
        count = 0
        frame = 0
        while success:
            if count % self.sample_every == 0:
                height, width = image.shape[:2]
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                if video_to_label is not None:
                    cv2.imwrite(f"{self.output_folder}/{label}/frame_{video_name}_{frame}.jpg", image)
                else:
                    cv2.imwrite(f"{self.output_folder}/frame_{video_name}_{frame}.jpg", image)

                frame += 1
            success, image = vidcap.read()
            count += 1
        video_num += 1
        vidcap.release()

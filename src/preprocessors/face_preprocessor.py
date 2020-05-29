import zipfile
import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from os.path import isfile, join
import face_recognition
import pickle


class FacePreprocessor:
    """
    Extract the faces from the videos.
    Faces are stored in a flat directory structure (no categorical hierarchy)

    Processes a video by extracting the faces from each frame and creating a
    list of list of faces which is then saved as a pickled object.

    NOTE: Faces are not guaranteed to be the same across frames.
          eg. 'face 1' in frame 1 may not be the same as 'face 1' in frame 2

    Pickle Object Format:
        [
            [
                [frame 1, face 1],
                [frame 1, face 2],
                [frame 1, face 3]
            ],
            [
                [frame 2, face 1],
                [frame 2, face 2]
            ],
            ...
        ]

    """

    def __init__(self, video_folder, output_folder, output_file=None, is_zip=True, height=320, width=480,
                 sample_every=10, max_workers=32):
        """
        @param video_folder          The folder where the list of videos frames are stored. If
                                     `is_zip` is set to True, this should be a single zip
                                     file containing the video frames. Paths can either by a local
                                     folder or a GDrive mounted path.
        @param output_folder         The local output path where the preprocessed files will be stored for
                                     further preprocessing can be done
        @param output_file           If not none, the output_folder will be zipped up and stored at this location
        @param is_zip                If set to True, the `video_folder` will be unzipped prior to accessing
        @param height         Height of the extracted video frames
        @param width          Width of the extracted video frames
        @param sample_every   The frames to skip.
        @param max_workers    The number of workers to use to parallelize work.
        """
        self.is_zip = is_zip
        self.video_folder = video_folder
        self.output_folder = output_folder
        self.output_file = output_file
        print(
            f"Video Preprocessor created with is_zip = {is_zip}, video_folder = {video_folder} , output_folder = {output_folder}, output_file = {output_file}")

        self.height = height
        self.width = width
        self.sample_every = sample_every
        self.max_workers = max_workers
        print(f"Frames will be created with height = {height} , width = {width} , sample_every = {sample_every}")

    def preprocess(self):
        tmp_output_folder = ""
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

        # Create output folder
        Path(f"{self.output_folder}/faces-pickle/").mkdir(parents=True, exist_ok=True)

        # Process each video by extracting the frames in a multi-threaded fashion
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            videos = next(os.walk(tmp_output_folder))[2]
            print(f"Found {len(videos)} videos")
            video_num = 1

            for video_name in videos:
                future = executor.submit(self.process_video, self.process_audio
                m
                tmp_output_folder, video_name, video_num, len(videos))
                futures.append(future)
                video_num += 1

        cv2.destroyAllWindows()

        print("***** Submitted all tasks *****")
        with open(f"{self.output_folder}/summary.txt", 'w') as f:
            f.write(
                f"video_name,face_image_name,frame_number,face_number,total_frames,fps,video_width,video_height,top,right,bottom,left\n")
            for future in futures:
                out_arr = future.result()
                for out in out_arr:
                    f.write(out)
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

    def process_video(self, tmp_output_folder, video_name, video_num, total_videos):
        """
        Processes a video by extracting the faces
        """
        vidcap = cv2.VideoCapture(join(tmp_output_folder, video_name))
        print(f"Processing video {video_num}/{total_videos} with name {video_name} \n")

        input_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))

        metadata = []
        faces_all_frames = []
        success, image = vidcap.read()
        count = 0
        frame = 0
        while success:
            if count % self.sample_every == 0:
                height, width = image.shape[:2]
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

                # Convert from BGR color (OpenCV) to RGB color (face_recognition)
                rgb_image = image[:, :, ::-1]

                # Find all the faces in the current frame of video
                face_locations = face_recognition.face_locations(rgb_image)
                faces = []
                face_num = 0
                # Display the results
                for top, right, bottom, left in face_locations:
                    # Draw a box around the face
                    faces.append(image[top:bottom, left:right, :].copy())
                    metadata.append(
                        f"{video_name},frame-{count}.face-{face_num}.jpg,{count},{face_num},{input_length},{fps},{frame_width},{frame_height},{top},{right},{bottom},{left}\n")
                    face_num += 1
                faces_all_frames.append(faces)

                frame += 1
            success, image = vidcap.read()
            count += 1
        video_num += 1
        vidcap.release()

        with open(f"{self.output_folder}/faces-pickle/{video_name}.pkl", "wb") as f_out:
            pickle.dump(faces_all_frames, f_out)
        return metadata

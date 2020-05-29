import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import tensorflow as tf
from os.path import isfile, join
import pickle
import glob
import subprocess
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
import numpy as np
import openl3
import soundfile as sf


class AudioPreprocessor:
    """
    Extract the audio from the videos.
    Faces are stored in a flat directory structure (no categorical hierarchy)
    """

    def __init__(self, video_folder, output_folder, output_file=None, label_path=None, is_zip=True, sample_every=10,
                 hop_size=0.5, max_len=5):
        """
        @param video_folder          The folder where the list of videos frames are stored. If
                                     `is_zip` is set to True, this should be a single zip
                                     file containing the video frames. Paths can either by a local
                                     folder or a GDrive mounted path.
        @param output_folder         The local output path where the preprocessed files will be stored for
                                     further preprocessing can be done
        @param output_file           If not none, the output_folder will be zipped up and stored at this location
        @param label_path            The path of the .txt file containing the class labels matched to the sample name.
        @param is_zip                If set to True, the `video_folder` will be unzipped prior to accessing
        - hop_size: The frame collection rate
        @param sample_every the frames to skip.
        """
        self.is_zip = is_zip
        self.video_folder = video_folder
        self.output_folder = output_folder
        self.output_file = output_file
        self.hop_size = hop_size
        self.max_len = max_len
        self.label_path = label_path
        print(
            f"Video Preprocessor created with is_zip = {is_zip}, video_folder = {video_folder} , output_folder = {output_folder}, output_file = {output_file}")

        self.sample_every = sample_every
        print(f"Frames will be created with hop_size = {hop_size}")

    def preprocess(self, batch_size=16):
        """
        Outputs: Writes to disk the openl3 embedding pickle object for each sample.
        Optionally, it will output the entire matched X and Y numpy pickle objects if label path is provided
        -
        """

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

        Path(f"{self.output_folder}/audio-pickle/").mkdir(parents=True, exist_ok=True)

        # Strip the audio from video and store as .wav file
        video_files = sorted(glob.glob(tmp_output_folder + '/*.mp4'))
        video_files_split = np.array_split(np.asarray(video_files), len(video_files) // batch_size)

        target_labels = []

        if self.label_path is not None:
            targets = []
            target_labels = np.genfromtxt(self.label_path, delimiter=' ', dtype='str')

        sr = 0
        all_x = []

        maxlen = int(self.max_len // self.hop_size + 1)

        for i in range(0, len(video_files_split)):

            audio_reads = []

            for f in video_files_split[i]:
                newname = os.path.basename(f)
                output_wav_file = newname + 'extracted_audio.wav'
                ffmpeg_extract_audio(f, "/tmp/" + output_wav_file)
                if self.label_path is not None:
                    target_index = np.where(target_labels[:, 0] == newname[:-4])[0]
                    target_index = int(target_index)
                    target = int(target_labels[:, 1][target_index]) - 1
                    targets.append(target)
                audio_read, sr = sf.read("/tmp/" + output_wav_file)
                audio_reads.append(audio_read)
                print(f"Reading file {output_wav_file} ...")

            X_arr, ts_list = openl3.get_audio_embedding(audio_reads, sr, batch_size=15, hop_size=self.hop_size)

            X = tf.keras.preprocessing.sequence.pad_sequences(X_arr, maxlen=maxlen)
            X = np.asarray(X, dtype='float32')

            if i == 0:
                all_x = X
                all_x = np.asarray(all_x, dtype='float32')
            else:
                all_x = np.concatenate((all_x, X), axis=0)

            print(all_x.shape)

        for f in video_files:
            file_name = os.path.basename(f)
            with open(f"{self.output_folder}/audio-pickle/{file_name}-openl3.pkl", "wb") as f_out:
                pickle.dump(all_x[i], f_out)

        if self.label_path is not None:
            with open(f"{self.output_folder}/audio-pickle-all-X-openl3.pkl", "wb") as f_out:
                pickle.dump(all_x, f_out)

            targets = np.asarray(targets)
            with open(f"{self.output_folder}/audio-pickle-all-Y-openl3.pkl", "wb") as f_out:
                pickle.dump(targets, f_out)

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

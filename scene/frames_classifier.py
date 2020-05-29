import zipfile
import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from os.path import isfile, join
import tensorflow as tf
import tempfile
import shutil

from generators.frame_generator import DataGenerator


class FramesClassifier:
    """
    Classifies sentiment based on frames extracted from video clips
    """

    def __init__(self, frames_folder, model_location=None, is_test=None, is_zip=True, frames_to_use=12, batch_size=16):
        """
        @param frames_folder  The folder where the list of frames are stored. If
                              `is_zip` is set to True, this should be a single zip
                              file containing the frames. Paths can either by a local
                              folder or a GDrive mounted path.
        @param model_location The pre-trained model to perform predictions
        @param is_test        If set to True, we assume that `frames_folder` contains a flat
                              list of videos. If False, we assume that `frames_folder` first
                              contains subdirectories corresponding to category labels.
        @param is_zip         If set to True, the `frames_folder` will be unzipped prior to accessing
        @param frames_to_use  The number of frames to use per video
        @param batch_size     The batch size used to feed into the model evaluation
        """
        self.is_zip = is_zip
        self.frames_folder = frames_folder
        self.is_test = is_test
        self.model_location = model_location
        self.frames_to_use = frames_to_use
        self.batch_size = batch_size
        print(f"FramesClassifier created with is_zip = {is_zip}, frames_folder = {frames_folder} , is_test = {is_test} , model_location = {model_location}")

    def predict(self, layer=None):
        folder = self.unzip_folder()
        generator = DataGenerator(folder, is_test=self.is_test, frames_to_use=self.frames_to_use, batch_size=self.batch_size)

        if "https://" in self.model_location or "http://" in self.model_location:
            downloaded_model_path = tf.keras.utils.get_file("frame-classifier", self.model_location)
            model = tf.keras.models.load_model(downloaded_model_path)
        else:
            model = tf.keras.models.load_model(self.model_location)
        if layer is not None:
            print(f"Customizing model by returning layer {layer}")
            model = tf.keras.models.Model(model.input, model.get_layer(layer).output)
        return model.predict(generator)

    def summary(self):
        model = tf.keras.models.load_model(self.model_location)
        model.summary()

    def evaluate(self):
        if self.is_test:
            print("Evaluation cannot be done in test-mode")
            return

        folder = self.unzip_folder()
        generator = DataGenerator(folder, is_test=self.is_test, frames_to_use=self.frames_to_use, batch_size=self.batch_size)
        model = tf.keras.models.load_model(self.model_location)
        return model.evaluate(generator)

    def unzip_folder(self):
        if self.is_zip:
              # Unzips files to a temp directory
              tmp_output_folder = "frames_tmp"
              if os.path.exists(tmp_output_folder) and os.path.isdir(tmp_output_folder):
                  print("Removing existing dir...")
                  shutil.rmtree(tmp_output_folder)

              print(f"Unzipping files to temp dir {tmp_output_folder}...")
              Path(f"{tmp_output_folder}").mkdir(parents=True, exist_ok=True)
              with zipfile.ZipFile(self.frames_folder, 'r') as zip_ref:
                  zip_ref.extractall(tmp_output_folder)
              print("Finished unzipping files")
        else:
            tmp_output_folder = self.frames_folder
            print("Skipping unzipping files as input is a folder")
        return tmp_output_folder


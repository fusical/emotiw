# -*- coding: utf-8 -*-
"""audio-final-classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wHOFUEoOQsLsixPYWLnuwZweSSP9xkVN
"""

import zipfile
import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from os.path import isfile, join
import tensorflow as tf
import tempfile
import shutil
import numpy as np
from sklearn.preprocessing import Normalizer


class AudioClassifier:
    """
    Classifies sentiment based on audio extracted from videos
    """

    def __init__(self, audio_folder , model_location=None, is_test=True, batch_size=32):
        """
        @param audio_folder   The folder where the arrays of processed audio embeddings are stored. If
                              ends with .zip, this should be a single zip
                              file containing the embeddings. Paths can either be accessed by a local
                              folder or a GDrive mounted path.
        @param model_location The pre-trained model to perform predictions
        @param is_test        If set to True, we assume that we are testing . If false, the evaluate
                              function will return a score.
        @param batch_size     The batch size used to feed into the model evaluation
        """
        self.audio_folder = audio_folder
        self.is_test = is_test
        self.model_location = model_location
        self.batch_size = batch_size
        print(f"AudioClassifier created with audio_folder = {audio_folder} , is_test = {is_test} , model_location = {model_location}")

    def predict(self, layer=None):
        folder = self.unzip_folder()
        X = np.load(os.path.join(folder, 'audio-pickle-all-X-openl3.pkl'), allow_pickle=True)

        if "https://" in self.model_location or "http://" in self.model_location:
            downloaded_model_path = tf.keras.utils.get_file("audio-classifier", self.model_location)
            model = tf.keras.models.load_model(downloaded_model_path)
        else:
            model = tf.keras.models.load_model(self.model_location)

        if layer is not None:
            print(f"Customizing model by returning layer {layer}")
            model = tf.keras.models.Model(model.input, model.get_layer(layer).output)

        normalizer = Normalizer()
        for i in range(0, X.shape[0]):
            X[i] = normalizer.fit_transform(X[i])

        return model.predict(X, batch_size=self.batch_size)

    def summary(self):
        model = tf.keras.models.load_model(self.model_location)
        model.summary()

    def evaluate(self):
        if self.is_test:
            print("Evaluation cannot be done in test-mode")
            return

        folder = self.unzip_folder()
        X = np.load(os.path.join(folder, 'audio-pickle-all-X-openl3.pkl'), allow_pickle=True)
        Y = np.load(os.path.join(folder, 'audio-pickle-all-Y-openl3.pkl'), allow_pickle=True)
        model = tf.keras.models.load_model(self.model_location)
        normalizer = Normalizer()
        for i in range(0,X.shape[0]):
            X[i] = normalizer.fit_transform(X[i])

        return model.evaluate(X , Y)

    def unzip_folder(self):
        if self.audio_folder.endswith(".zip"):
            # Unzips files to a temp directory
            tmp_output_folder = "audio_tmp"
            if os.path.exists(tmp_output_folder) and os.path.isdir(tmp_output_folder):
                print("Removing existing dir...")
                shutil.rmtree(tmp_output_folder)

            print(f"Unzipping files to temp dir {tmp_output_folder}...")
            Path(f"{tmp_output_folder}").mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.audio_folder, 'r') as zip_ref:
                zip_ref.extractall(tmp_output_folder)
            print("Finished unzipping files")
        else:
            tmp_output_folder = self.audio_folder
            print("Skipping unzipping files as input is a folder")
        return tmp_output_folder

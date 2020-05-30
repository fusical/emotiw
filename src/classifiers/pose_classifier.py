import tensorflow as tf
from ..generators.pose_generator import PoseDataGenerator
import tensorflow as tf

from .utils import unzip_folder


class PoseClassifier:
    """
    Classifies sentiment based on poses extracted from video frames
    """

    def __init__(self, pose_folder, model_location=None, is_test=None, frames_to_use=12, batch_size=32):
        """
        @param pose_folder    The folder where the list of poses are stored. If
                              ends with .zip, this should be a single zip
                              file containing the poses. Paths can either by a local
                              folder or a GDrive mounted path.
        @param model_location The pre-trained model to perform predictions
        @param is_test        If set to True, we assume that `pose_folder` contains a flat
                              list of videos. If False, we assume that `pose_folder` first
                              contains subdirectories corresponding to category labels.
        @param frames_to_use  The number of frames to use per video
        @param batch_size     The batch size used to feed into the model evaluation
        """
        self.pose_folder = pose_folder
        self.is_test = is_test
        self.model_location = model_location
        self.frames_to_use = frames_to_use
        self.batch_size = batch_size
        print(f"PoseClassifier created with pose_folder = {pose_folder} , is_test = {is_test} , model_location = {model_location}")

        if "https://" in self.model_location or "http://" in self.model_location:
            downloaded_model_path = tf.keras.utils.get_file("pose-classifier", self.model_location)
            self.model = tf.keras.models.load_model(downloaded_model_path)
        else:
            self.model = tf.keras.models.load_model(self.model_location)

    def predict(self, layer=None):
        """
        Performs sentiment classification prediction on preprocessed pose files
        @param layer: If None, performs normal sentiment classification.
                      If not None, returns the values from the intermediate layers.
        :return:
            - The model prediction result
            - The video file names for each of the rows returned in model.predict
              (without the .mp4 suffix)
        """
        folder = unzip_folder(self.pose_folder, "pose_tmp")
        generator = PoseDataGenerator(folder, is_test=self.is_test, frames_to_use=self.frames_to_use, batch_size=self.batch_size)

        if layer is not None:
            print(f"Customizing model by returning layer {layer}")
            model = tf.keras.models.Model(self.model.input, self.model.get_layer(layer).output)
        else:
            model = self.model

        # Determine the order of samples that the generator gave to the model
        samples = map(lambda x: x.split(".mp4")[0].split("frame_")[1], generator.video_names)
        return model.predict(generator), list(samples)

    def summary(self):
        """
        Summarizes the pre-trained model
        """
        self.model.summary()

    def evaluate(self):
        """
        Evaluates the pose files on the pre-trained model
        return: The evaluation results
        """
        if self.is_test:
            print("Evaluation cannot be done in test-mode")
            return

        folder = unzip_folder(self.pose_folder, "pose_tmp")
        generator = PoseDataGenerator(folder, is_test=self.is_test, frames_to_use=self.frames_to_use, batch_size=self.batch_size)

        return self.model.evaluate(generator)

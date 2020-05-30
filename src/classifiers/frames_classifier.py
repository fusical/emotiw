import tensorflow as tf

from .utils import unzip_folder
from ..generators.frame_generator import FramesDataGenerator


class FramesClassifier:
    """
    Classifies sentiment based on frames extracted from video clips
    """

    def __init__(self, frames_folder, model_location=None, is_test=None, frames_to_use=12, batch_size=16):
        """
        @param frames_folder  The folder where the list of frames are stored. If
                              `is_zip` is set to True, this should be a single zip
                              file containing the frames. Paths can either by a local
                              folder or a GDrive mounted path.
        @param model_location The pre-trained model to perform predictions
        @param is_test        If set to True, we assume that `frames_folder` contains a flat
                              list of videos. If False, we assume that `frames_folder` first
                              contains subdirectories corresponding to category labels.
        @param frames_to_use  The number of frames to use per video
        @param batch_size     The batch size used to feed into the model evaluation
        """
        self.frames_folder = frames_folder
        self.is_test = is_test
        self.model_location = model_location
        self.frames_to_use = frames_to_use
        self.batch_size = batch_size
        print(f"FramesClassifier created with frames_folder = {frames_folder} , is_test = {is_test} , model_location = {model_location}")

    def predict(self, layer=None):
        """
        Performs sentiment classification prediction on preprocessed frames files
        @param layer: If None, performs normal sentiment classification.
                      If not None, returns the values from the intermediate layers.
        return:
            - The model prediction result
            - The video file names for each of the rows returned in model.predict
              (without the .mp4 suffix)
        """
        folder = unzip_folder(self.frames_folder, "frames_tmp")
        generator = FramesDataGenerator(folder, is_test=self.is_test, frames_to_use=self.frames_to_use, batch_size=self.batch_size)

        if "https://" in self.model_location or "http://" in self.model_location:
            downloaded_model_path = tf.keras.utils.get_file("frame-classifier", self.model_location)
            model = tf.keras.models.load_model(downloaded_model_path)
        else:
            model = tf.keras.models.load_model(self.model_location)
        if layer is not None:
            print(f"Customizing model by returning layer {layer}")
            model = tf.keras.models.Model(model.input, model.get_layer(layer).output)

        # Determine the order of samples that the generator gave to the model
        samples = map(lambda x: x.split(".mp4")[0], generator.video_names)

        return model.predict(generator), samples

    def summary(self):
        """
        Summarizes the pre-trained model
        """
        model = tf.keras.models.load_model(self.model_location)
        model.summary()

    def evaluate(self):
        """
        Evaluates the frames files on the pre-trained model
        return: The evaluation results
        """
        if self.is_test:
            print("Evaluation cannot be done in test-mode")
            return

        folder = unzip_folder(self.frames_folder, "frames_tmp")
        generator = FramesDataGenerator(folder, is_test=self.is_test, frames_to_use=self.frames_to_use, batch_size=self.batch_size)
        model = tf.keras.models.load_model(self.model_location)
        return model.evaluate(generator)


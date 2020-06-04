from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer
import os, cv2
import numpy as np

class FaceClassifier:
  """
  Classifies sentiment based on facial expressions extracted from videos
  """
  def __init__(self, face_folder, is_test, model_location, batch_size):
      """
        @param face_folder   The folder where the arrays of processed facial expression embeddings are stored. If
                              ends with .zip, this should be a single zip
                              file containing the embeddings. Paths can either be accessed by a local
                              folder or a GDrive mounted path.
        @param model_location The pre-trained model to perform predictions of labels
        @param is_test        If set to True, we assume that we are testing . If false, the evaluate
                              function will return a score.
        @param batch_size     The batch size used to feed into the model evaluation
        """
        self.face_folder = face_folder
        self.is_test = is_test
        self.model_location = model_location
        self.batch_size = batch_size
        print(f"FacesClassifier created with face_folder = {face_folder} , is_test = {is_test} , model_location = {model_location}")
        if "https://" in self.model_location or "http://" in self.model_location:
            downloaded_model_path = tf.keras.utils.get_file("audio-classifier", self.model_location)
            self.model = load_model(downloaded_model_path)
        else:
            self.model = load_model(self.model_location)

  def load_example_data(self):
      self.X_train = np.load(self.face_folder + "xtrain.npy")
      self.Y_train = np.load(self.face_folder + "ytrain.npy").astype(int)
      self.X_val   = np.load(self.face_folder + "xval.npy")
      self.Y_val   = np.load(self.face_folder + "yval.npy").astype(int)

  def predict(self, layer=None):
      """
        Performs sentiment classification prediction on preprocessed audio files
        @param layer: If None, performs normal sentiment classification.
                      If not None, returns the values from the intermediate layers.
        return:
            - The model prediction result
            - The video file names for each of the rows returned in model.predict
              (without the .mp4 suffix)
        """
        folder = unzip_folder(self.face_folder, "audio_tmp")
        X = np.load(os.path.join(folder, 'audio-pickle-all-X-openl3.pkl'), allow_pickle=True)


        if layer is not None:
            print(f"Customizing model by returning layer {layer}")
            model = tf.keras.models.Model(self.model.input, self.model.get_layer(layer).output)
        else:
            model = self.model

        normalizer = Normalizer()
        for i in range(0, X.shape[0]):
            X[i] = normalizer.fit_transform(X[i])

        # The original pre-processing created the X array using the sorted order of the video files
        audio_pickles = sorted(next(os.walk(os.path.join(self.face_folder, "audio-pickle")))[2])
        samples = map(lambda x: x.split(".mp4")[0], audio_pickles)

        return model.predict(X, batch_size=self.batch_size), list(samples)

  def load_model(self, best_model_filepath):
      self.model = load_model(best_model_filepath)

  # Inputs (N,22): min, max, mean of all facial hidden features, # of faces in frame
  # Ouputs (3): frame sentiment
  def init_model(self):
      layers = [tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2())),
          tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
      ]

      self.model = tf.keras.Sequential(layers)
      optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
      self.model.compile(optimizer=optimizer,
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])


  def train(self, epochs=200, val_split=0.1, save_path = None):
      """
      - Outputs
        1. Trained model -- saves the model as an .h5 file to the specified path
      """

      es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300)
      mc = ModelCheckpoint(save_path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

      _, history = self.model.fit(self.X_train, self.y_train, epochs=epochs,
      batch_size=self.batch_size, validation_data=(self.X_val, self.Y_val))

      return self.model, history

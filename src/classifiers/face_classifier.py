from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer
import os, cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path


class FaceClassifier:
  """
  Classifies sentiment based on facial expressions extracted from videos
  """
  def __init__(self, face_folder, is_test, model_location=None, batch_size=32):
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

      self.load_data()

      print(f"FacesClassifier created with face_folder = {face_folder} , is_test = {is_test} , model_location = {model_location}")
      if self.model_location is not None:
          if "https://" in self.model_location or "http://" in self.model_location:
              downloaded_model_path = tf.keras.utils.get_file("face-classifier", self.model_location)
              self.model = load_model(downloaded_model_path)
          else:
              self.model = load_model(self.model_location)
      else:
          self.model = self.init_model()

  def load_data(self):
      print("load data")
      self.X = np.load(os.path.join(self.face_folder, "faces-fer-X.npy"))
      if os.path.exists(os.path.join(self.face_folder, "faces-fer-Y.npy")):
          self.Y = np.load(os.path.join(self.face_folder, "faces-fer-Y.npy")).astype(int)[:, 1]
          self.Y = tf.keras.utils.to_categorical(self.Y, num_classes=3)

  def predict(self, layer=None):
      if layer is not None:
          print(f"Customizing model by returning layer {layer}")
          model = tf.keras.models.Model(self.model.input, self.model.get_layer(layer).output)
      else:
          model = self.model
      path = os.path.join(self.face_folder, "faces-fer-Y.npy")
      y = np.load(path, allow_pickle=True)
      samples = map(lambda x: str(x), y[:, 0])
      return model.predict(self.X, batch_size=self.batch_size), list(samples)

  def summary(self):
      self.model.summary()

  def load_model(self, best_model_filepath):
      return load_model(best_model_filepath)

  # Inputs (N,22): min, max, mean of all facial hidden features, # of faces in frame
  # Ouputs (3): frame sentiment
  def init_model(self):
      def create_model(inputs):
          x = inputs
          x = layers.BatchNormalization()(x)
          x = layers.Conv1D(16, 3, activation='relu')(x)
          x = layers.Flatten()(x)
          #x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2()))(inputs)
          x = layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())(x)
          #x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())(x)
          x = layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())(x)

          # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)

          optimizer = tf.keras.optimizers.Adam()
          model = tf.keras.Model(inputs=inputs, outputs=x)
          model.compile(optimizer=optimizer,
                        loss = 'categorical_crossentropy',
                        metrics=['accuracy'])
          return model

      inputs = tf.keras.Input(shape=(self.X.shape[1], self.X.shape[2]))
      model = create_model(inputs)
      return model

  def train(self, faces_val_folder, epochs=20, save_path=None):
      """
      - Outputs
        1. Trained model -- saves the model as an .h5 file to the specified path
      """
      self.X_val = np.load(os.path.join(faces_val_folder, "faces-fer-X.npy"))
      self.Y_val = np.load(os.path.join(faces_val_folder, "faces-fer-Y.npy")).astype(int)[:, 1]
      self.Y_val = tf.keras.utils.to_categorical(self.Y_val, num_classes=3)

      es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300)
      mc = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', mode='max', verbose=1, save_best_only=True)

      history = self.model.fit(self.X, self.Y, 
                                epochs=epochs,
                                batch_size=self.batch_size, 
                                callbacks=[es, mc],
                                validation_data=(self.X_val, self.Y_val))

      return self.model, history










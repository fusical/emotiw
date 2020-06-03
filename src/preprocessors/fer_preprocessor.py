import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import os
from os.path import isfile, join
import pickle
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import numpy as np

class FerPreprocessor:
    """
    Run FER on all faces.
    """
    def __init__(self, faces_folder, output_folder, model_path, output_file=None,
                 max_workers=32):

        self.faces_folder = faces_folder
        self.output_folder = output_folder
        self.output_file = output_file
        self.max_workers = max_workers
        self.model_path = model_path
        print(
            f"FER Preprocessor created with faces_folder = {faces_folder}, output_folder = {output_folder}, output_file = {output_file}")

    def preprocess(self):
        tmp_input_folder = ""
        if self.faces_folder.endswith(".zip"):
            # Unzips files to a temp directory
            tmp_input_folder = self.output_folder.rstrip('/') + "_tmp"
            print(f"Unzipping files to temp dir {tmp_input_folder}...")
            Path(f"{tmp_input_folder}").mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.faces_folder, 'r') as zip_ref:
                zip_ref.extractall(tmp_input_folder)
            print("Finished unzipping files")
        else:
            tmp_input_folder = path.join(self.faces_folder, "faces-pickle")
            print("Skipping unzipping files as input is a folder")

        Path(f"{self.output_folder}/fer-pickle/").mkdir(parents=True, exist_ok=True)
        fer_model = load_model(self.model_path)

        videos = next(os.walk(tmp_input_folder))[2]
        for i in range(len(videos)):
            self.process_video(fer_model, tmp_input_folder, \
                               videos[i], i+1, len(videos))

        # Process each face
        # futures = []
        # with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #     videos = next(os.walk(tmp_input_folder))[2]
        #     print(f"Found {len(videos)} videos")
        #     video_num = 1

        #     for video_name in videos:
        #         future = executor.submit(self.process_video, fer_model, tmp_input_folder, \
        #                                 video_name, video_num, len(videos))
        #         futures.append(future)
        #         video_num += 1

        print("***** Submitted all tasks *****")
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

    def process_video(self, fer_model, tmp_input_folder, pkl_name, video_num, \
                      total_videos, extract=False):
        print(f"Processing video {video_num}/{total_videos} with name {pkl_name}")
        max_frames = 12
        
        if extract:
            fer_model = Model(inputs=fer_model.input, outputs=fer_model.layers[-3].output)
            _, H, W, C = fer_model.output_shape
            X_train = np.empty((num_videos, max_frames, H*W*C*3 + 1))
        else:
            X_train = np.empty((max_frames, 22))

        pkl_path = path.join(tmp_input_folder, pkl_name)
        with open(pkl_path, "rb") as f_in:
          frames = pickle.load(f_in)
          for frame_i in range(min(len(frames), max_frames)):
              faces = frames[frame_i]
              if len(faces) != 0:
                  fer_scores = self.run_fer(faces, fer_model, extract) # (N,7)
                  max_face_scores = np.amax(fer_scores, axis=0).flatten() # (1,7)
                  min_face_scores = np.amin(fer_scores, axis=0).flatten() # (1,7)
                  mean_face_scores = np.mean(fer_scores, axis=0).flatten() # (1,7)
                  num_faces = np.array([len(faces)])
                  X_train[frame_i, :] = np.concatenate((max_face_scores, min_face_scores, mean_face_scores, num_faces))

        with open(f"{self.output_folder}/fer-pickle/{pkl_name}", "wb") as f_out:
            pickle.dump(X_train, f_out)
  
    def run_fer(self, faces, fer_model, extract=False):
      N = len(faces)
      if extract:
        _, H, W, C = fer_model.output_shape
        fer_scores = np.empty((N,H,W,C))
      else:
        fer_scores = np.empty((N,7))

      for i in range(N):
        X = cv2.resize(faces[i], (48,48))
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        X = cv2.normalize(X,0,255)
        X = X.reshape(1,48,48,1)

        fer_scores[i,:] = fer_model.predict(X)
        #rloc = make_relative_loc(f"{vid_name}.mp4",framei*10,facei)
        #X_train[0,framei,facei,:7] = y_pred_prob
        #X_train[0,framei,facei,7:] = rloc

      return fer_scores
  

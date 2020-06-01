import zipfile
import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
from os.path import isfile, join
import face_recognition
import pickle
from tensorflow.keras import Model


class FerPreprocessor:
    """
    Run FER on all faces.
    """
    def __init__(self, faces_folder, output_folder, output_file=None, height=320, width=480,
                 sample_every=10, max_workers=32):

        self.faces_folder = faces_folder
        self.output_folder = output_folder
        self.output_file = output_file
        print(
            f"FER Preprocessor created with faces_folder = {faces_folder} , output_folder = {output_folder}, output_file = {output_file}")

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
            tmp_input_folder = self.faces_folder
            print("Skipping unzipping files as input is a folder")

        # Create output folder
        Path(f"{self.output_folder}/fer-pickle/").mkdir(parents=True, exist_ok=True)

        # Process each face
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            videos = next(os.walk(tmp_input_folder))[2]
            print(f"Found {len(videos)} videos")
            video_num = 1

            for video_name in videos:
                future = executor.submit(self.process_video, tmp_input_folder, video_name, video_num, len(videos))
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

    def process_videos(self, tmp_input_folder, video_name, video_num, total_videos):
        print(f"Processing video {video_num}/{total_videos} with name {video_name} \n")

        # run fer on each face
        self.run_fer(video_name)

        with open(f"{self.output_folder}/faces-pickle/{video_name}.pkl", "wb") as f_out:
            pickle.dump(faces_all_frames, f_out)
        return metadata
  
def run_fer(faces, fer_model, extract=False):
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

def preprocess(Y, pkl_folder, model_path, max_frames=12, extract=False):
  '''
  Train our network on all frames in all videos.
  '''
  fer_model_soa = load_model(model_path)
  # FACES_FOLDER = "/content/drive/My Drive/cs231n-project/datasets/emotiw/train/faces-pickle"
  num_videos = Y.shape[0]
  if extract:
    fer_model_soa = Model(inputs=fer_model_soa.input, outputs=fer_model_soa.layers[-3].output)
    _, H, W, C = fer_model_soa.output_shape
    X_train = np.empty((num_videos, max_frames, H*W*C*3 + 1))
  else:
    X_train = np.empty((num_videos, max_frames, 22))
  Y_train = np.zeros(num_videos)
  print(X_train.shape)
  print(Y_train.shape)

  for vid_i in range(num_videos):
    vid_name = Y.iloc[vid_i]['Vid_name']
    print(f"{vid_name}.mp4... ", end='')
    with open(f"{pkl_folder}/{vid_name}.mp4.pkl", "rb") as f_in:
        frames = pickle.load(f_in)
        for frame_i in range(min(len(frames), max_frames)):
          faces = frames[frame_i]
          if len(faces) != 0:
            fer_scores = run_fer(faces, fer_model_soa, extract) # (N,7)
            max_face_scores = np.amax(fer_scores, axis=0).flatten() # (1,7)
            min_face_scores = np.amin(fer_scores, axis=0).flatten() # (1,7)
            mean_face_scores = np.mean(fer_scores, axis=0).flatten() # (1,7)
            num_faces = np.array([len(faces)])
            X_train[vid_i, frame_i, :] = np.concatenate((max_face_scores, min_face_scores, mean_face_scores, num_faces))
        Y_train[vid_i] = Y[Y['Vid_name'] == vid_name]['Label'] - 1
  np.save(f"{self.output_folder}/xtrain.npy", X_train)
  np.save(f"{self.output_folder}/ytrain.npy", Y_train)
  

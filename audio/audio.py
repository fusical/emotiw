from tensorflow.keras.models import load_model
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
import os
from SliceAudio import slice_audio
import arffToNp

# add option for soft vs hard
def predict(mp4_filepath, best_model_filepath):
    """
    Outputs:
    - A tuple with predictions for each class (positive, neutral, negative)
    """

    model = fer_model()
    model.load_model(best_model_filepath)
    return model.predict(mp4_filepath)

class audio_model:
    def __init__(self):
        self.model = ()
        return

    def predict(self, mp4_filepath):
        self.preprocess(mp4_filepath)
        X = cv2.imload("test/happy.jpg")
        X = cv2.resize(X, (48,48))
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)

        return self.model.predict(img)
        #return (0.1,0.2,0.7)

    def load_model(self, best_model_filepath):
        self.model = load_model(best_model_filepath)
        return

    def train(self, mp4_filepaths):
        # train the model
        # self.model = ....
        return

    def preprocess(self, mp4_filepath):

        """
        Outputs:
        - A numpy array with dimensions (m,n). m is the units in time dependent on the audio splice rate.
            n is the number of features from the openSMILE library.
        """

        output_wav_file = os.path.dirname(mp4_filepath) + 'extracted_audio.wav'
        mp4_filename = os.path.basename(mp4_filepath)

        # Strip the audio from video and store as .wav file
        ffmpeg_extract_audio(mp4_filepath, output_wav_file)

        # splice the audio files into 2 seconds with 100 ms sliding window.
        # 30 kHz sampling rate
        slice_audio([output_wav_file] , channels=2 , outformat='wav', rate=30000 , slide=100)

        # Walk through each sliced file and get the openSmile features from that file

        out_fn = os.path.join(opensmile, mp4_filename[-4] + '-openSMILE-features.arff')
        csv_path =

        for root, dirs, files in os.walk(os.path.dirname(mp4_filepath) , topdown=False):
            for name in files:
                # Ignore the original input file
                if 'extracted_audio.wav' not in name:


                    #TO DO: Run bash in python
                    os.system("cd 'opensmile-2.3.0' ; inst/bin/SMILExtract -C config/IS13_ComParE.conf -I \"$in_fn\" -O \"$out_fn\" -N $name")

        # Convert .arff to .csv
        all_timepoints_feature_array = arffToNp.convert(out_fn)


        return all_timepoints_feature_array

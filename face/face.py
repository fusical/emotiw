from tensorflow.keras.models import load_model
import cv2


# add option for soft vs hard
def predict(mp4_filepath, best_model_filepath):
    """
    Outputs:
    - A tuple with predictions for each class (positive, neutral, negative)
    """
    
    model = fer_model()
    model.load_model(best_model_filepath)
    return model.predict(mp4_filepath)

class fer_model:
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
        # break out the faces from video (Vincent/Tom's code)
        return

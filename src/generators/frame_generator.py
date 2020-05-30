import numpy as np
import random
import cv2
from os import listdir
from os.path import isfile, join
import tensorflow as tf


class FramesDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras generator for frames
    """

    def __init__(self, dir, batch_size=32, frames_to_use=12, is_test=False, shuffle=True, height=320, width=480):
        self.frames_to_use = frames_to_use
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.is_test = is_test
        self.dir = dir
        if is_test:
            self.shuffle = False
        else:
            self.shuffle = shuffle
        self.classes = self.find_classes()
        self.video_names, self.video_map, self.video_to_class, self.num_samples, self.min_frames = self.find_samples()
        self.on_epoch_end()
        if self.is_test:
            print(f"Found {self.num_samples} frames belonging to {len(self.video_names)} videos (test-mode).")
        else:
            print(
                f"Found {self.num_samples} frames belonging to {len(self.video_names)} videos belonging to {len(self.classes)} classes.")
        print(f"Min frames determined to be {self.min_frames}")

    def find_classes(self):
        if self.is_test:
            return []
        else:
            category_folders = [f for f in listdir(self.dir) if not isfile(join(self.dir, f))]
            return sorted(list(set(category_folders)))

    def find_samples(self):
        num_samples = 0
        min_frames = -1
        video_map = {}
        vid_to_cat = {}
        if self.is_test:
            category_folders = [self.dir]
        else:
            category_folders = [f for f in listdir(self.dir) if not isfile(join(self.dir, f))]
        for category_folder in category_folders:
            if self.is_test:
                cat_path = category_folder
            else:
                cat_path = join(self.dir, category_folder)
            frames = [f for f in listdir(cat_path) if isfile(join(cat_path, f))]
            for frame in frames:
                # frame = frame_101_7.mp4_8.jpg
                frame_arr = frame.split(".mp4_")
                vid_name = frame_arr[0]
                if vid_name not in video_map:
                    video_map[vid_name] = []
                    vid_to_cat[vid_name] = category_folder
                video_map[vid_name].append(frame)

            for k in video_map.keys():
                # make sure the frames for each video are in sorted order
                video_map[vid_name] = sorted(video_map[vid_name])
                if min_frames == -1 or len(video_map[vid_name]) < min_frames:
                    min_frames = len(video_map[vid_name])

        return list(video_map.keys()), video_map, vid_to_cat, len(vid_to_cat), min_frames

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        video_names = self.video_names[index * self.batch_size:(index + 1) * self.batch_size]
        num_frames = self.min_frames if self.frames_to_use == -1 else self.frames_to_use
        X = np.zeros((len(video_names), num_frames, self.height, self.width, 3), dtype=np.uint8)
        y = []
        i = 0
        for vid in video_names:
            j = 0
            for frame in self.video_map[vid]:
                if self.is_test:
                    frame_path = join(self.dir, frame)
                else:
                    frame_path = join(join(self.dir, self.video_to_class[vid]), frame)

                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X[i, j, :, :, :] = img
                j += 1
                if j >= num_frames:
                    break

            if self.is_test:
                y.append(0)
            else:
                y.append(int(self.video_to_class[vid]) - 1)
            i += 1
        y = np.array(y)
        return X, tf.keras.utils.to_categorical(y, num_classes=len(self.classes))

    def on_epoch_end(self):
        if self.is_test == False and self.shuffle == True:
            np.random.shuffle(self.video_names)

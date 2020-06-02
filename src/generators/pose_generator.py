import numpy as np
from os import listdir
from os.path import isfile, join
import json
import tensorflow as tf


class PoseDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras generator for raw pose keypoint data.

    Only body keypoints are extracted and normalized.
    """

    def __init__(self, keyframe_dir, batch_size=32, frames_to_use=-1, is_test=False, shuffle=True):
        self.frames_to_use = frames_to_use
        self.batch_size = batch_size
        self.keyframe_dir = keyframe_dir
        if is_test:
            self.shuffle = False
        else:
            self.shuffle = shuffle
        self.is_test = is_test
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
            category_folders = [f for f in listdir(self.keyframe_dir) if not isfile(join(self.keyframe_dir, f))]
            return sorted(list(set(category_folders)))

    def find_samples(self):
        """
        """
        num_samples = 0
        min_frames = -1
        video_map = {}
        vid_to_cat = {}

        if self.is_test:
            category_folders = [self.keyframe_dir]
        else:
            category_folders = [f for f in listdir(self.keyframe_dir) if not isfile(join(self.keyframe_dir, f))]
        print(category_folders)
        for category_folder in category_folders:
            if self.is_test:
                cat_path = category_folder
            else:
                cat_path = join(self.keyframe_dir, category_folder)
            frames = [f for f in listdir(cat_path) if isfile(join(cat_path, f))]
            for frame in frames:
                # frame = frame_7_6.mp4_0_keypoints.json
                frame_arr = frame.split(".mp4_")
                vid_name = frame_arr[0]
                if vid_name not in video_map:
                    video_map[vid_name] = []
                    vid_to_cat[vid_name] = category_folder
                video_map[vid_name].append(frame)

            for k in video_map.keys():
                # make sure the frames for each video are in sorted order
                video_map[k] = sorted(video_map[k],
                                      key=lambda x: x.split(".mp4_")[0] + x.split(".mp4_")[1].split("_keypoints")[0].zfill(3))
                if min_frames == -1 or len(video_map[k]) < min_frames:
                    min_frames = len(video_map[k])

        return list(video_map.keys()), video_map, vid_to_cat, len(vid_to_cat), min_frames

    def get_body_joints(self, x):
        body_parts = [
            1,  # neck       --
            2,  # r shoulder
            3,  # r elbow
            4,  # r wrist
            5,  # l shoulder
            6,  # l elbow
            7,  # l wrist
            9,  # r hip
            10,  # r knee
            11,  # r ankle
            12,  # l hip
            13,  # l knee
            14,  # l ankle   --
        ]
        body_parts_xy = []
        for b in body_parts:
            body_parts_xy.append(b * 3)
            body_parts_xy.append(b * 3 + 1)
        return x[body_parts_xy]

    def normalize(self, x_input):
        # Separate original data into x_list and y_list
        lx = []
        ly = []
        N = len(x_input)
        i = 0
        while i < N:
            lx.append(x_input[i])
            ly.append(x_input[i + 1])
            i += 2
        lx = np.array(lx)
        ly = np.array(ly)

        # Get rid of undetected data (=0)
        non_zero_x = []
        non_zero_y = []
        for i in range(int(N / 2)):
            if lx[i] != 0:
                non_zero_x.append(lx[i])
            if ly[i] != 0:
                non_zero_y.append(ly[i])
        if len(non_zero_x) == 0 or len(non_zero_y) == 0:
            return np.array([0] * N)

        # Normalization x/y data according to the bounding box
        origin_x = np.min(non_zero_x)
        origin_y = np.min(non_zero_y)
        len_x = np.max(non_zero_x) - np.min(non_zero_x)
        len_y = np.max(non_zero_y) - np.min(non_zero_y)
        x_new = []
        for i in range(int(N / 2)):
            if (lx[i] + ly[i]) == 0:
                x_new.append(-1)
                x_new.append(-1)
            else:
                x_new.append((lx[i] - origin_x) / len_x)
                x_new.append((ly[i] - origin_y) / len_y)
        return x_new

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        video_names = self.video_names[index * self.batch_size:(index + 1) * self.batch_size]
        num_frames = self.min_frames if self.frames_to_use == -1 else self.frames_to_use
        X = np.zeros((len(video_names), num_frames, 13 * 2 + 1), dtype=np.float64)
        y = []
        i = 0
        for vid in video_names:
            j = 0
            for frame in self.video_map[vid]:
                if self.is_test:
                    keypoint_file = join(self.keyframe_dir, frame)
                else:
                    keypoint_file = join(join(self.keyframe_dir, self.video_to_class[vid]), frame)
                with open(keypoint_file) as json_file:
                    keypoint_data = json.load(json_file)

                    # Extract some features from the keypoint data like averaging
                    arrs = []

                    for person in keypoint_data["people"]:
                        # Each person is assigned the label of the video
                        kp = np.array(person["pose_keypoints_2d"])
                        kp = self.get_body_joints(kp)
                        kp = self.normalize(kp)
                        arrs.append(kp)
                        # if i == 0 and j == 0:
                        #     print(kp)

                    if len(arrs) > 0:
                        arrs = np.array(arrs)
                        features = []
                        features.extend(np.average(arrs, axis=0).tolist())
                        features.append(len(keypoint_data["people"]))
                        features = np.array(features)
                        features[np.isnan(features)] = -1
                        X[i, j, :] = np.array(features)

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
        if self.shuffle == True:
            np.random.shuffle(self.video_names)

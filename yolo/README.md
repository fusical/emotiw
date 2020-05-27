This directory contains files relevant for YOLO v3.

File Descriptions (mostly from following this [tutorial](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/))
1. yolov3.weights: Pre-trained model weights that were trained using the DarkNet code base on the MSCOCO dataset
    1. Download Link: https://pjreddie.com/media/files/yolov3.weights
    2. Note, this file is actually around 230 MB so too large for GitHub. I stored the weights in our Google Drive folder here: https://drive.google.com/file/d/1f5FH_QAtvQP7W03_e0E3c3F2hRjKwgtF/view?usp=sharing.
2. yolo3_one_file_to_detect_them_all.py: provides the make_yolov3_model() function to create the model for us, and the helper function \_conv\_block() that is used to create blocks of layers.
    1. Note, I had to make one change related to this bug
        1. Replace `np.set_printoptions(threshold=np.nan)` with `np.set_printoptions(threshold=sys.maxsize)`
3. metadata_all.json: Output of running YOLOv3 on training videos. Outputs, for each video, objects detected, confidence scores, and bounding box locations
4. yolov3_object_detection.ipynb: Notebook going through example of running YOLOv3 on the images to obtain outputs. Produces metadata_all.json.

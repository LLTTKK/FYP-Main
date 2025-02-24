Final Year Project: Vehicle Traffic Analysis Using Machine Learning and Deep Learning
================================================================================

Project Overview
---------------
This Final Year Project (FYP) implements a traffic analysis system combining machine learning and deep learning approaches. The system utilizes transfer learning with YOLOv5 for object detection and Logistic Regression for classification.

Key Components
-------------
1. Object Detection: Custom-trained YOLOv5 model (yolov5_updated.pt)
2. Classification: Logistic Regression model (LR.joblib)
3. Video Processing: Frame extraction every 10 seconds

System Requirements
-----------------
- Python (recommended: latest stable version)
- Required libraries (install via pip):
  - YOLOv5 dependencies
  - scikit-learn
  - opencv-python
  - numpy

Usage Instructions
----------------
Run the system using the following command format:

python FYP_LR.py --video_path [path of the video input] --save_dir [path for saving images per 10 seconds] --weights [path of YOLOv5 custom model weights] --lr_model [path of Logistic Regression Model]


Parameters:
- video_path: Path to the input video file
- save_dir: Directory where frames will be saved (every 10 seconds)
- weights: Path to the custom YOLOv5 model weights (yolov5_updated.pt)
- lr_model: Path to the trained Logistic Regression model (LR.joblib)

Project Structure
---------------
- FYP_LR.py: Main application script
- yolov5_updated.pt: Custom trained YOLOv5 weights
- LR.joblib: Trained Logistic Regression model
- dataset.zip: Training dataset

License
-------
This project is licensed under the MIT License.





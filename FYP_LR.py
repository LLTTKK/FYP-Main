#!/usr/bin/env python
# coding: utf-8

# Output:
# 1. Full or not --> txt 1.png full 2.png not full
# 2. sliced image (original)
# 3. recognized 1. cor // 2. cor on photo (new photo: original + box)
# 4. Decidsion algorithm: based on full or not to stop or not
# 5. Simulation:
# 
# NOW:
# INPUT: mp4 vid
# slice mp4 video into 10 sec segment
# pass to detection
# output cor
# overlap / draw cor on oringinal and ouput
# full or not
# decision algorithm: not stop or stop
# 
# give output

import torch
import torch.nn as nn
import cv2
from PIL import Image
import os
import argparse
import numpy as np
from joblib import load

# Create the parser
parser = argparse.ArgumentParser(description='Process a video and save the results.')

# Add the arguments
parser.add_argument('--video_path', type=str, help='The path to the video file.')
parser.add_argument('--save_dir', type=str, help='The directory where the results will be saved.')
parser.add_argument('--weights', type=str, help='The path to the weights file.')
parser.add_argument('--lr_model', type=str, help='The path to the trained Logistic Regression Model.')

# Parse the arguments
args = parser.parse_args()

# Path to your video file
video_path = args.video_path

# Directory where you want to save the results
save_dir = args.save_dir

# Path to the weights file
weights_path = args.weights

lr_path=args.lr_model
# Load the custom YOLOv5 model weight
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)


def is_elevator_full(results, model_path=lr_path, confidence_threshold=0.9):
    # Initialize logistic regression model
    model = load(lr_path)

    # Get the number of people detected
    num_people = len(results.xyxy[0])

    # Reshape the data to a 2D array
    x_unseen = np.array([[num_people]])

    # Use the model to predict the probability
    probabilities = model.predict_proba(x_unseen)

    # The probabilities variable is a 2D array where the first element of each entry is the probability of the sample being in the 0 class (elevator not full) and the second element is the probability of the sample being in the 1 class (elevator full).
    prob_elevator_full = probabilities[0][1]


    print(f"The predicted probability of the elevator being full for {x_unseen[0][0]} person is {prob_elevator_full* 100}%")

    # If the prediction is over the confidence threshold, the elevator is considered full
    is_full = prob_elevator_full > confidence_threshold

    return is_full




def process_video(video_path, model, save_dir, interval_seconds=10):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame interval based on the desired interval in seconds
    frame_interval = int(fps * interval_seconds)

    frame_count = 0
    while cap.isOpened():
        # Read the next frame from the video
        ret, frame = cap.read()

        if ret:
            # If we've reached the frame interval, process the frame
            if frame_count % frame_interval == 0:
                # Convert the frame to a PIL image
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)

                # Perform inference on the image
                results = model(img_pil)
                
                # Change the current working directory to the save directory
                os.chdir(save_dir)


                #Print the results and the number of person in the elevator
                num_people = len(results.xyxy[0])
                print("\n", results.xyxy)  # Print predictions: [xmin, ymin, xmax, ymax, confidence, class]
                print("There is ", num_people, " person in the Elevator")
                
                # Determine whether the elevator is full or not
                full = is_elevator_full(results)
                print("Elevator is full" if full else "Elevator is not full", "\n")
                
                # Save the results with the default filename
                results.save()

            frame_count += 1
        else:
            break

    # Release the video file
    cap.release()
    cv2.destroyAllWindows()



# Call the function
process_video(video_path, model, save_dir)



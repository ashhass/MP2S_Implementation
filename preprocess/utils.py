import os
import cv2
import numpy as np


def extract_frames(training_path, frame_path, video):
    """
    Extracts frames from a video and saves them as jpg files
    :param training_path: path to the training videos
    :param frame_path: path to the frames
    :return: None
    """
    # loop through all the videos
    cap = cv2.VideoCapture(training_path)

    # initialize a counter variable
    count = 0

    # loop through all the frames
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(f'{frame_path}/{video}_{count}.jpg', frame) 
            count += 1
            print(count)
        else:
            break

    cap.release() 


def extract_flowMap(frame_list):  # returns stacked flow map 

    flow = None
    flowMap_list = []
    for i in range(len(frame_list)):
        prev_frame_gray = cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2GRAY)
        next_frame_gray = cv2.cvtColor(frame_list[i + 1], cv2.COLOR_BGR2GRAY) if i + 1 < len(frame_list) else prev_frame_gray

        if flow is None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0)

        flow_norm = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
        flow_norm = flow_norm.astype('uint8')

        flowMap_list.append(flow_norm[:,:,0])
        prev_frame_gray = next_frame_gray

    return np.stack(flowMap_list)  

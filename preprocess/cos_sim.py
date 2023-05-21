''' 
    1. Convert flow maps to 1D vectors
    2. Compute similarity between vectors from different classes
    3. Calculate the average overall similarity
'''
import os
import sys
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F 
from utils import extract_frames, extract_flowMap

video_train_path = '/y/ayhassen/anomaly_detection/ucf/Anomaly-Videos-Part_2/Anomaly-Videos-Part-2/Fighting/' 
frame_train_path_1 = '/y/ayhassen/anomaly_detection/ucf/Anomaly-Videos-Part_2/Anomaly-Videos-Part-2/Fighting/frames'
frame_train_path_2 = '/y/ayhassen/anomaly_detection/ucf/Normal_Videos_for_Event_Recognition/Normal_Videos_for_Event_Recognition/frames'



# compare motion patterns   
anomaly_frame_list = []
normal_frame_list = []
sim_list = [] 
num = 0

for count in range(800):
    if num < 8:
        frame_anomaly = np.array(Image.open(f'{frame_train_path_1}/{os.listdir(frame_train_path_1)[count]}').convert('RGB'))
        anomaly_frame_list.append(frame_anomaly)

        frame_normal = np.array(Image.open(f'{frame_train_path_2}/{os.listdir(frame_train_path_2)[count]}').convert('RGB'))
        normal_frame_list.append(frame_normal) 
    
    elif num == 8:
        anomaly = extract_flowMap(anomaly_frame_list)
        normal = extract_flowMap(normal_frame_list)
        # similarity = F.cosine_similarity(torch.from_numpy(anomaly).view(1, -1).float(), torch.from_numpy(normal).view(1, -1).float()).item()
        print(normal.shape)
        similarity = torch.nn.MSELoss()(torch.from_numpy(anomaly).view(1, -1).float(), torch.from_numpy(normal).view(1, -1).float()).item()
        num = 0
        print(similarity)
        sim_list.append(similarity) 
        
    elif num > 8:
        anomaly_frame_list = []
        normal_frame_list = []

    print(f'{count} out of {len(os.listdir(frame_train_path_1))}')
    num += 1

print(f'Average: {sum(sim_list)/len(sim_list)}') # average similarity between anomaly and normal motion patterns 
import math
from cv2 import resize
import os
import pdb
import sys
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import torchvision

class CustomDataset(Dataset):

    
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(f'{self.image_dir}')   

    def __len__(self):
        return (len(self.images) - 20) // 100

    def __getitem__(self, index):

        img_path_list = [os.path.join(f'{self.image_dir}', self.images[index + x]) for x in range(8)] 
        image = np.array(Image.open(img_path_list[0]).convert('RGB')) 

        image_list = [] 
        for images in img_path_list:
            image_list.append(np.array(Image.open(images).convert('RGB')))

        flowMap = self.extract_flowMap(image_list) 

        if self.transform is not None:
            augmentations = self.transform(image=image, flowMap=flowMap) 
            flowMap = augmentations["flowMap"] 

        return flowMap
 

    def extract_flowMap(self, frame_list):  # returns stacked flow map 

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
    
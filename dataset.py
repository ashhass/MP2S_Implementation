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

class CustomDataset(Dataset):

    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(f'{self.image_dir}')   

    def __len__(self):
        return len(self.images) 

    def __getitem__(self, index):

        img_path_list = [os.path.join(f'{self.image_dir}', self.images[index + x]) for x in range(8)] 

        image_list = np.array(Image.open(img_path_list[x] for x in range(len(img_path_list))).convert('RGB'))  

        flowMap = self.extract_flowMap(image_list)

        if self.transform is not None:
            augmentations = self.transform(image=image, flowMap=flowMap)
            image = augmentations["image"]
            flowMap = augmentations["flowMap"]
        
        return image, flowMap 
 

    def extract_flowMap(self, frame_list):  # returns stacked flow map 

        flow = None
        flowMap_list = []
        for i in range(len(frame_list)):
            prev_frame_gray = cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2GRAY)
            next_frame_gray = cv2.cvtColor(frame_list[i + 1] , cv2.COLOR_BGR2GRAY)

            if flow is None:
                flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            else:
                flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0)

            flow_norm = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
            flow_norm = flow_norm.astype('uint8')

            flowMap_list.append(flow_norm[:,:,0])
            prev_frame_gray = next_frame_gray

        return np.hstack(flowMap_list) 
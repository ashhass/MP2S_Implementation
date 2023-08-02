import torch
import io
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from models.unsup_model import Conv_AE_LSTM
from models.sup_model import ResNet
from PIL import Image


def get_model1(): 
     # define and load unsupervised model here
     model = Conv_AE_LSTM()
     checkpoint = torch.load('checkpoints/modelv1.pth', map_location=torch.device('cpu'))
     model.load_state_dict(checkpoint['state_dict'])
     model.eval()

     return model 

def get_model2(): 
     # define and load supervised model here
     model = ResNet(depth=18, num_classes=2, num_frames=1)
     checkpoint =  torch.load('checkpoints/modelv2.tar', map_location=torch.device('cpu'))
     model.load_state_dict(checkpoint['state_dict'])
     model.eval()

     return model


def get_tensor(image): 
    my_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224))
                    ])
    # image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0) 



def extract_flowMap(frame_list):

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

        return np.hstack(flowMap_list) 
import torch
import torchvision
import torch.nn as nn
import cv2 
import numpy as np
import matplotlib.pyplot as plt
# from model.convlstm import ConvLSTM

class Conv_AE_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(10,10), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(6,6), stride=2),
            nn.ReLU(),  
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64,  kernel_size=(6,6), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=(10,10), stride=2),
            nn.ReLU(), 

        )

    def forward(self, x):
        # print(f'X:  {x.shape}')
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded 
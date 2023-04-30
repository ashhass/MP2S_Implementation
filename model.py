
'''   VERSION 01
    1. Construct bare minimum convolutional autoencoder layers
    2. Add convolutional LSTM layers
    3. Modularize your code
    4. Comment throughout 
    5. Visualize encoded features
'''

'''    VERSION 02
    1. CONSTRUCT AN ENSEMBLE NETWORK OF ANOMALY DETECTION USING OPTIC FLOW AND ACTION RECOGNITION NETWORK (to help reduce the false negatives) (MAJORITY VOTING)
    2. TEST THAT ON LOCAL VIDEO COLLECTION

'''
import torch
import torchvision
import torch.nn as nn
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from lstm import ConvLSTM

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
            nn.ConvTranspose2d(128, 64, kernel_size=(6,6), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=(10,10), stride=2),
            nn.ReLU(), 

        )

    def forward(self, x):
        print(f'X:  {x.shape}')
        # encoded = self.encoder(x)

        # shape, filter_size, input_channels, num_features, num_layers
        convlstm = ConvLSTM((856, 480), 3, 1, 10, 2)
        hidden_state = convlstm.init_hidden(batch_size=1)
        encoded = convlstm(x, hidden_state)

        print(f'Encoded: {encoded.shape}')
        decoded = self.decoder(encoded)

        return decoded

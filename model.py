'''
    1. Construct bare minimum convolutional autoencoder layers
    2. Add convolutional LSTM layers
    3. Modularize your code
    4. Comment throughout 
    5. Visualize encoded features
'''
import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded



# flow_maps = [cv2.imread(f'./flow/flow{x}.png') for x in range(7)]
# image = np.stack(flow_maps)
# image = torch.from_numpy(image).float().permute(0, 3, 1, 2)

# model = Conv_AE_LSTM()
# output = model(image) 

# torchvision.utils.save_image(output, './out.png') 
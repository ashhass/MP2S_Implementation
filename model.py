
'''   VERSION 01
    1. Construct bare minimum convolutional autoencoder layers
    2. Add convolutional LSTM layers
    3. Modularize your code
    4. Comment throughout 
    5. Visualize encoded features
'''

'''    VERSION 02
    1. CONSTRUCT AN ENSEMBLE NETWORK OF ANOMALY DETECTION USING OPTIC FLOW AND ACTION RECOGNITION NETWORK (to help reduce the false negatives) (MAJPRITY VOTING)
    2. DEFINE A SET OF SEQUENTIAL ACTIONS THAT WOULD LEAD TO A POSSIBLE INCIDENT (combine that with the optic flow method to detect anomalies early on)
    3. TEST THAT ON LOCAL VIDEO COLLECTION

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
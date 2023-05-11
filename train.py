import sys  
import torch 
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm, trange
import torch.nn as nn
import torch.optim as optim
from model.model import Conv_AE_LSTM
from util import (
    load_checkpoint,
    save_checkpoint,
    get_loaders, 
) 
from loss import MS_SSIM
import matplotlib.pyplot as plt 
import numpy as np
import timm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Hyperparameters 
LEARNING_RATE = 1e-6
IMAGE_WIDTH = 856
IMAGE_HEIGHT = 480
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 10
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = False 


TRAIN_DIR = '/y/ayhassen/anomaly_detection/shanghaitech/training_set/frames'
VAL_DIR = '/y/ayhassen/anomaly_detection/shanghaitech/training_set/frames'   # modify to actual validation path


def main():
    transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ), 
            ToTensorV2(),
        ], 
    ) 
    
    model = Conv_AE_LSTM().to(DEVICE) 
    # loss_fn = MS_SSIM(channel=1)  # set channel to 1 considering our input is grayscale
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(  
        TRAIN_DIR,
        VAL_DIR,
        BATCH_SIZE,
        transform,
        NUM_WORKERS,
        PIN_MEMORY, 
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("./checkpoints/modelv1.pth"), model) 
 
    scaler = torch.cuda.amp.GradScaler()
    epoch = 0
    for epoch in range(NUM_EPOCHS):
        count = 0
        running_loss = 0.0
        loss_values = []
        for batch_idx, targets in enumerate(train_loader):
            targets = targets.unsqueeze(1).to(device=DEVICE, dtype=torch.float32) / 255
            targets = torch.reshape(targets, (targets.shape[2], targets.shape[0], targets.shape[1], targets.shape[4], targets.shape[3]))
            # targets = torch.reshape(targets, (targets.shape[1], targets.shape[0], targets.shape[3], targets.shape[2]))
            # forward
            with torch.cuda.amp.autocast(): 
                predictions = model(targets).to(dtype=torch.float32) 
                print(predictions.shape, targets.shape)
                loss = loss_fn(predictions, targets[0,:,:,:,:]) 
                writer.add_scalar("Loss(train) - Epoch", loss, epoch) 

            # backward 
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer) 
            scaler.update() 
            running_loss += loss.item()
            loss_values.append(running_loss)

            if batch_idx % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 10))
                running_loss = 0.0

            print(f'NUMBER OF ELEMENTS IN THE TRAIN_LOADER : {count}') 
            count+=1

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }

        loss = loss_fn(predictions, targets) 
        writer.add_scalar("Loss(train) - Epoch", loss, epoch) 
    

        save_checkpoint(checkpoint, "./checkpoints/modelv1.pth") 
        print(f'EPOCH NUMBER :  {epoch}')
        epoch+=1

writer.flush() 


if __name__ == "__main__":
    main() 
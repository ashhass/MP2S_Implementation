import torch 
import torchvision
from dataset import CustomDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


def save_checkpoint(state, filename):  
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])



def get_loaders(
    train_dir,
    val_dir,
    batch_size,
    transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CustomDataset(
        image_dir=train_dir,
        transform=transform, 
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CustomDataset( 
        image_dir=val_dir,
        transform=transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory, 
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):

    # implement the reconstruction metric here

    model.train()


def save_predictions_as_imgs(
    loader, model, folder, device="cuda"
):
    model.eval()
    count = 1
    for idx, y in enumerate(loader): 
        if count <= 10:
            y = y.unsqueeze(0).to(device='cuda', dtype=torch.float32) / 255
            with torch.no_grad():
                preds = torch.sigmoid(model(y))      

                torchvision.utils.save_image(
                    preds, f"{folder}/pred_{idx}.png" 
                )
                torchvision.utils.save_image(
                    y, f"{folder}/actual_{idx}.png"   
                )    # this should be the actual optic flow map and not the video frame
                
                count+=1
                print(count)

    model.train() 
    

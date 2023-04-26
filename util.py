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

def endpoint_error(loader, model, device="cuda"):
    epe = 0
    model.eval()
    with torch.no_grad():
        for i, y_true in enumerate(loader):
            y_true = y_true.unsqueeze(0).to(device=device, dtype=torch.float32) / 255
            y_pred = torch.sigmoid(model(y_true))

            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()

            error_vector = np.linalg.norm(y_true - y_pred, axis=-1)
            epe += np.mean(error_vector)
            print(i, epe / len(loader))
            

    print(f'End point Error: {epe / len(loader)}')
    model.train() 

def mse(loader, model, device='cuda'):
    model.eval()
    with torch.no_grad():
        for y_true in loader:
            y_true = y_true.unsqueeze(0).to(device=device, dtype=torch.float32) / 255
            y_pred = torch.sigmoid(model(y_true))

            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()

            mse = np.mean(np.power(y_true - y_pred, 2), axis=1)
            reconstruction_error = np.mean(mse)
            print(reconstruction_error)

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
    

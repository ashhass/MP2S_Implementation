from util import *  
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import Conv_AE_LSTM

LEARNING_RATE = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 35
NUM_WORKERS = 1
IMAGE_HEIGHT = 480 
IMAGE_WIDTH = 850 
PIN_MEMORY = True
LOAD_MODEL = True

TRAIN_DIR ='/y/ayhassen/anomaly_detection/shanghaitech/training_set/frames'
VAL_DIR ='/y/ayhassen/anomaly_detection/shanghaitech/training_set/frames'   


transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
) 

train_loader, val_loader = get_loaders( 
        TRAIN_DIR,
        VAL_DIR,
        BATCH_SIZE,
        transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )


model = Conv_AE_LSTM().to(DEVICE)

if LOAD_MODEL:
    load_checkpoint(torch.load("./checkpoints/modelv1.pth"), model)

endpoint_error(train_loader, model, device=DEVICE) 
# save_predictions_as_imgs(
#             train_loader, model, folder='./test', device=DEVICE
#             ) 
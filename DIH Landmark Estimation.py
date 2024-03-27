import os, json, cv2, random, numpy as np, matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import loggers as pl_loggers


import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F


import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl

import transforms, utils, engine, train
from utils import collate_fn

from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms as T

#grab all data from json and port that to target
#return full image, return full targets
#integrate boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format.
#integrate keypoints (FloatTensor[N, K, 3]): for each one of the N objects, it contains the K keypoints in [x, y, visibility] format, defining the object. visibility=0 means that the keypoint is not visible.
#integrate labels (Int64Tensor[N]): the label for each bounding box. 0 always represents the background class.

class MyDataset(Dataset):
    def __init__(self, directory, transform=None, filter=False):
        """
        Initialize the MyDataset class.

        Args:
            directory (str): The directory path where the dataset is located.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
                Defaults to None.
            filter (bool, optional): Flag indicating whether to use the filtered annotations directory. Defaults to False.
        """
        super(MyDataset, self).__init__()

        self.image_dir = os.path.join(directory, "image")

        if filter:
            self.annos_dir = os.path.join(directory, "filtered")
        else:
            self.annos_dir = os.path.join(directory, "annos")

        self.length = len(os.listdir(self.annos_dir))

        self.lookup = os.listdir(self.annos_dir)
        for i in range(len(self.lookup)):
            self.lookup[i] = self.lookup[i].split(".")[0]

        self.transform = transform

    def __getitem__(self, idx):
        idx_f = self.lookup[idx]

        image_path = os.path.join(self.image_dir, str(idx_f) + ".jpg")
        image = read_image(image_path, ImageReadMode.RGB) / 255.0
        image_size = image.shape[1:]  # Assuming image shape is [C, H, W]

        with open(os.path.join(self.annos_dir, str(idx_f) + ".json")) as f:
            annos = json.load(f)

        if self.transform is not None:
            image = self.transform(image)
            
        keypoints_raw = annos["landmarks"]

        # Reshape keypoints and discard visibility if present
        keypoints = np.array(keypoints_raw).reshape(-1, 3)
        keypoints = keypoints[:, :2].astype(np.float32)  # Keep only x and y coordinates

        # Normalize keypoints to [0, 1] range based on image dimensions
        keypoints[:, 0] = keypoints[:,0]/image_size[1]  # Normalize x coordinates by width
        keypoints[:, 1] = keypoints[:,1]/image_size[0]  # Normalize y coordinates by height

        # Convert to tensor
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        return image, keypoints

    def __len__(self):
        return self.length
    
class LitNetwork(pl.LightningModule):
    def __init__(self,num_k=25,batch_size=1):
        super(LitNetwork, self).__init__()

        self.model = keypointrcnn_resnet50_fpn(num_keypoints=num_k,weights_backbone=ResNet50_Weights.IMAGENET1K_V1)

        self.loss_func = nn.MSELoss()

        #self.val_ap = torchmetrics.AveragePrecision(task="multiclass",num_classes=num_k)
        self.b = batch_size

    def forward(self, image, targets):
        print(image.shape)
        print(targets)
        x = self.model(image, targets)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, data, batch_idx):
        image, targets = data[0], data[1]

        out = self.forward(image,targets)
        out = out[0]["keypoints"]
        loss = self.loss_func(out[0,:,:2], targets)

        self.log("train_loss",loss,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.b,sync_dist=True)
        return loss
    
    def validation_step(self, val_data, batch_idx):
        image, targets = val_data[0], val_data[1]

        loss_dict = self.forward(image,targets)
        #print(out[0])
        out = out[0]["keypoints"]
        #print(out.shape)
        #print(targets.shape)
        print(loss_dict)
        loss = sum(loss for loss in loss_dict.values())#self.loss_func(out[0,:,:2], targets)

        self.log("val_loss",loss,batch_size=self.b,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        #self.val_ap(out, targets)
        #self.log("val_ap",self.val_ap,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        return None
    
    #never ending sanity check

def train_network(workers=8):

    train_dataset = MyDataset("C:\\Users\\crisz\\Documents\\ECU Classes\\CSCI Graduate\\Thesis\\DeepFashion2\\train",filter=True)
    validation_dataset = MyDataset("C:\\Users\\crisz\\Documents\\ECU Classes\\CSCI Graduate\\Thesis\\DeepFashion2\\validation",filter=True)

    b = 1
    train_loader = DataLoader(train_dataset,batch_size=b,num_workers=workers,persistent_workers=False,shuffle=True)
    val_loader = DataLoader(validation_dataset,batch_size=b,num_workers=workers,persistent_workers=False)

    model = LitNetwork(25,b)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    logger = pl_loggers.TensorBoardLogger(save_dir="my_logs")
    logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

    device = "gpu" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU

    trainer = pl.Trainer(max_epochs=200, accelerator=device, callbacks=[checkpoint], logger=logger, num_sanity_val_steps=0)
    torch.set_float32_matmul_precision('high')
    trainer.fit(model,train_loader,val_loader)
        


train_network(workers=8)

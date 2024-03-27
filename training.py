import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader

from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
from torchvision.io import read_image,ImageReadMode
from torchvision import transforms

import numpy as np
import os
import json

class MyDataset(Dataset):
    def __init__(self, directory, filter=False):
        super(MyDataset, self).__init__()

        self.image_dir = os.path.join(directory,"image")

        if filter:
            self.annos_dir = os.path.join(directory,"filtered")
        else:
            self.annos_dir = os.path.join(directory,"annos")

        self.length = len(os.listdir(self.annos_dir))

        self.lookup = os.listdir(self.annos_dir)
        for i in range(len(self.lookup)):
            self.lookup[i] = self.lookup[i].split(".")[0]

        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


    def __getitem__(self, idx):
        
        idx_f = self.lookup[idx]

        image = read_image(os.path.join(self.image_dir,str(idx_f)+".jpg"),ImageReadMode.RGB)/255
        with open(os.path.join(self.annos_dir,str(idx_f)+".json")) as f:
            annos = json.load(f)

        image = self.transform(image)
        keypoints_raw = annos["landmarks"]

        # TODO Properly Process Landmarks
        keypoints = np.array(keypoints_raw).reshape(-1,3)
        keypoints = keypoints[:,:2]

        # Convert to tensor
        keypoints = torch.tensor(keypoints,dtype=torch.float32)

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

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, data, batch_idx):
        image, targets = data[0], data[1]

        out = self.forward(image)
        out = out[0]["keypoints"]
        loss = self.loss_func(out[0,:,:2], targets)

        self.log("train_loss",loss,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.b,sync_dist=True)
        return loss
    
    def validation_step(self, val_data, batch_idx):
        image, targets = val_data[0], val_data[1]

        out = self.forward(image)
        #print(out[0])
        #print(targets)
        out = out[0]["keypoints"]
        #print(out.shape)
        #print(targets.shape)
        loss = self.loss_func(out[0,:,:2], targets)

        self.log("val_loss",loss,batch_size=self.b,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        #self.val_ap(out, targets)
        #self.log("val_ap",self.val_ap,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        return None



def train_network(workers=8):

    train_dataset = MyDataset("DeepFashion2/train/train/",filter=True)
    validation_dataset = MyDataset("DeepFashion2/validation/validation/",filter=True)

    b = 1
    train_loader = DataLoader(train_dataset,batch_size=b,num_workers=workers,persistent_workers=True,shuffle=True)
    val_loader = DataLoader(validation_dataset,batch_size=b,num_workers=workers,persistent_workers=True)

    model = LitNetwork(25,b)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    logger = pl_loggers.TensorBoardLogger(save_dir="my_logs")
    #logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

    device = "gpu" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU

    trainer = pl.Trainer(max_epochs=200, accelerator=device, callbacks=[checkpoint], logger=logger)
    torch.set_float32_matmul_precision('high')
    trainer.fit(model,train_loader,val_loader)
        


if __name__ == "__main__":

    train_network(workers=8)
 pl_loggers.TensorBoardLogger(save_dir="my_logs")
    #logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

    device = "gpu" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU

    trainer = pl.Trainer(max_epochs=200, accelerator=device, callbacks=[checkpoint], logger=logger)
    torch.set_float32_matmul_precision('high')
    trainer.fit(model,train_loader,val_loader)
        


if __name__ == "__main__":

    train_network(workers=8)
    device = "gpu" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU

    trainer = pl.Trainer(max_epochs=200, accelerator=device, callbacks=[checkpoint], logger=logger)
    torch.set_float32_matmul_precision('high')
    trainer.fit(model,train_loader,val_loader)
        


if __name__ == "__main__":

    train_network(workers=8)

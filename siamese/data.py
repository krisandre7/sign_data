from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
import torch

class SiameseData(Dataset):
  def __init__(self,train_csv=None,train_dir=None,transform=None):
    # used to prepare the labels and images path
    self.train_df = pd.read_csv(train_csv, names=["image1","image2","label"])
    self.train_dir = train_dir    
    self.transform = transform

  def __getitem__(self,index):
    # getting the image path
    image1_path = os.path.join(self.train_dir,self.train_df.iat[index,0])
    image2_path = os.path.join(self.train_dir,self.train_df.iat[index,1])
    
    # Loading the image
    img1: Image.Image = Image.open(image1_path)
    img2: Image.Image = Image.open(image2_path)
    img1 = img1.convert("L")
    img2 = img2.convert("L")

    # Apply image transormations
    if self.transform is not None:
      img1 = self.transform(img1) 
      img2 = self.transform(img2)

    label = torch.from_numpy(np.array([int(self.train_df.iat[index,2])],dtype=np.float32))
    return img1, img2, label

  def __len__(self):
    return len(self.train_df)

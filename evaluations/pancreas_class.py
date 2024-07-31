import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

import time
from cv2 import mean
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia as K
import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import glob
import natsort
from torchvision import transforms as T
from functools import partial

class pancreas(Dataset):
    def __init__(self,data_dir,state='train',im_size=(512,512)):
        if state=='train':
            self.data_dir_init = data_dir + '/'+'Train'
        else:
            self.data_dir_init=data_dir + '/'+'Test'

                #self.image_path_list=os.listdir(data_dir)
        self.img_list=[]
        self.class_list=[]
        self.file_list=natsort.natsorted(os.listdir(self.data_dir_init))
        num=0
        for file in self.file_list:
            self.img_path=self.data_dir_init+'/'+file+'/*'
            self.img_path_test=natsort.natsorted(glob.glob(self.img_path))
            self.img_list += self.img_path_test
            self.class_list += [num]*int(len(self.img_path_test))
            num += 1

        self.im_size=im_size
        self.transform=transforms.Compose(
                    [transforms.Grayscale(num_output_channels=1),transforms.Resize((512,512)),transforms.ToTensor(),transforms.Normalize(mean=0.5,std=0.5)]
                )
                #transforms.Resize((256,256))
       
    def __getitem__(self,idx):
        img_path=self.img_list[idx]
        #label=self.class_list[idx]
        img=Image.open(img_path)
        img=self.transform(img)
        return img


    def __len__(self):
        return len(self.img_list)
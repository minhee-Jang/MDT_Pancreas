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
import torchvision
import os
#import kornia as K
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
import random


class pancreas(Dataset):
    def __init__(self,data_dir,state='train',im_size=(512,512),label=None):
        if state=='train':
            self.data_dir_init = data_dir + '/'+'Train'
        else:
            self.data_dir_init=data_dir + '/'+'Test'

                #self.image_path_list=os.listdir(data_dir)
        self.img_list=[]
        self.class_list=[]
        self.label=list(label)
        
        self.file_list=natsort.natsorted(os.listdir(self.data_dir_init))
        num=0
        #random_index=random.sample(range(0,64289),5000)
        self.img_l=[]
        self.img_m=[]
        for file in self.file_list:
            self.img_path=self.data_dir_init+'/'+file+'/*'
            self.img_path_test=natsort.natsorted(glob.glob(self.img_path))
            (self.img_list).append(self.img_path_test)
            #self.img_list=self.img_list[random_index]
            (self.class_list).append([num]*int(len(self.img_path_test)))
            num += 1
        
        for i in range(3):
            num_cla=(self.label).count(i)
            (self.img_m).append(random.sample(self.img_list[i],num_cla))
        
        
        for m in range(len(self.label)):
            if self.label[m]==0:
                (self.img_l).append((self.img_m[0]).pop())
            elif self.label[m]==1:
                (self.img_l).append((self.img_m[1]).pop())
            else:
                (self.img_l).append((self.img_m[2]).pop())
        
        #for i in range(len(random_index)):
        #    (self.img_l).append(self.img_list[random_index[i]])
        #self.img_l=self.img_list[:5000]
        
        #print(len(self.img_list))
        
        self.im_size=im_size
        self.transform=transforms.Compose(
                    [transforms.Resize((512,512)),transforms.ToTensor(),transforms.Normalize(mean=0.5,std=0.5)]
                )
                #transforms.Resize((256,256))
                #,transforms.ToTensor()
       #transforms.Normalize(mean=0.5,std=0.5)
    def __getitem__(self,idx):
        img_path=self.img_l[idx]

        #label=self.class_list[idx]
        img=Image.open(img_path).convert('RGB')
        #img=np.array(img)
        #img=np.transpose(img,(2,0,1))
        #img=torch.from_numpy(img)
        #img=img.float()
        img=self.transform(img)

        #img=np.array(img)
        
        #img=np.transpose(img,(2,0,1))
       
        #img=np.resize(img,(3,512,512))
        #img=np.resize(img,(512,512,3))
        #img=torch.from_numpy(img)
        #img=self.transform(img)
        
        #img=img.permute(0,2,1)
        #img=img.contiguous()
        #img=img*2-1
        #img=torch.clamp(((img + 1) * 127.5),min=0,max=255).to(torch.uint8)
        
        #img=self.transform(img)
       
        #img=self.transform(img)
        #img=torch.as_tensor(img)
      
        return img


    def __len__(self):
        return len(self.img_l)
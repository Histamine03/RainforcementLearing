import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from PIL import Image
import torchvision.transforms as TRF
import torchvision.transforms.functional as TRF_F
import torchvision.utils as vutils
import math
import matplotlib.pyplot as plt
import re
import random
import cv2
import matplotlib.animation as animation
from torch.utils.data import DataLoader

def image_2_Tensor(root, image_size = (416, 416)):
    trans = TRF.Compose([
    TRF.Resize(image_size),
    TRF.ToTensor(),
    #TRF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])  
    data_folder = torchvision.datasets.ImageFolder(
    root = root, transform = trans
)   
    classes = data_folder.classes
    input_Tensor = []
    label_Tensor = []
    for data, lbl in data_folder:
        input_Tensor.append(data)
        label_Tensor.append(lbl)
    print(f"전체 데이터 수:  {len(label_Tensor)}")
    print("-----------------------")
    for idx, label_name in enumerate(classes):
        print(f"index_number {idx} | index_name {label_name} | number of data: {label_Tensor.count(idx)} 개 ")
    print("-----------------------")   
    print("데이터 로더 완료 !")
    return torch.cat(input_Tensor).type(torch.float32).view(-1,3,image_size[-1], image_size[-2]), torch.Tensor(label_Tensor)

def information(input_Tensor, lbl_Tensor):
    lbl_Tensor = list(map(int ,lbl_Tensor))
    print(f"전체 데이터 수 | {len(lbl_Tensor)} 개")
    print('-------------------------')
    classes = list(set(lbl_Tensor))

    for cls in classes:
        print(f'index number {cls} | {lbl_Tensor.count(cls)} 개')
    print('-------------------------')
    print(f'  image_size   | ({input_Tensor.shape[-1]} * {input_Tensor.shape[-2]})')

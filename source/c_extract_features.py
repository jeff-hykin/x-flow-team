'''
    step 1: !git clone https://github.com/mlmed/torchxrayvision.git
    step 2: !pip install torchxrayvision
    step 3: using a_image_preprocessing.py get_cropped_and_resized_images() to generate 224x224 images
    step 4: place this code under 'torchxrayvision/scripts/' fold
    step 5: run it to get feature csv
'''

import os,sys
sys.path.insert(0,"..")
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage
import pprint

import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms

import torchxrayvision as xrv
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('-fold_path', default='./new_train224/', type=str)
parser.add_argument('-weights', type=str,default="all")
parser.add_argument('-cuda', default=True, help='', action='store_true')

cfg = parser.parse_args()

img_items = os.listdir(cfg.fold_path)

model = xrv.models.DenseNet(weights=cfg.weights)
output_feat = np.empty((len(img_items), 1024), dtype=np.float32)
output_name = np.array(img_items)
output_name = np.expand_dims(output_name,axis=1)

for index, img_item in enumerate(img_items):
    img_path = cfg.fold_path + img_item
    img = skimage.io.imread(img_path)

    img = xrv.datasets.normalize(img, 255)
    img = img[:, :, 0]

    # Add color channel
    img = img[None, :, :]                  

    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        #   if cfg.cuda:
        #       img = img.cuda()
        #       model = model.cuda()
        
        feats = model.features(img)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        output_feat[index] = feats.cpu().detach().numpy().reshape(-1)
      
output = pd.DataFrame(np.concatenate((output_name, output_feat), axis=1))
output.to_csv('./feature_selections/pretrained.csv')
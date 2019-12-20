import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import copy
import tqdm
from PIL import Image

import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-o", "--dataFolder", required=True,help="data Folder")
args = vars(ap.parse_args())
dataPath = args['dataFolder']
# SVHN dataset
torchvision.datasets.SVHN(root=dataPath,
                          split='train', 
                          transform=None, 
                          target_transform=None, 
                          download=True);

torchvision.datasets.SVHN(root=dataPath',
                          split='test', 
                          transform=None, 
                          target_transform=None, 
                          download=True); 

torchvision.datasets.SVHN(root=dataPath,
                          split='extra', 
                          transform=None, 
                          target_transform=None, 
                          download=True); 
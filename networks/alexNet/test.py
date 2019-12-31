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
from tqdm import tqdm

import argparse

import yaml

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-o", "--dataFolder", required=True,help="data Folder")
#ap.add_argument("-e", "--numberEpoch", required=True,help="numberEpoch")
args = vars(ap.parse_args())
dataPath = args['dataFolder']

with open(r'parameter.yaml') as file:
    parmeters = yaml.load(file, Loader=yaml.FullLoader)

#print(parmeters)

'''
batch_size = 128
num_epochs = 5
num_classes = 10
learning_rate = 0.001
'''
batch_size = parmeters['batch_size']
num_epochs = parmeters['num_epochs']
num_classes = parmeters['num_classes']
learning_rate = parmeters['learning_rate']

modelSavedPath = parmeters['modelSavedPath']

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.Resize(227),
    transforms.ToTensor()
])

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# SVHN dataset
train_dataset = torchvision.datasets.SVHN(root=dataPath,
                          split='train', 
                          transform=data_transform, 
                          target_transform=None, 
                          download=True);

test_dataset = torchvision.datasets.SVHN(root=dataPath,
                          split='test', 
                          transform=data_transform, 
                          target_transform=None, 
                          download=True); 

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers = 6, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          num_workers = 6,
                                          shuffle=False)

total_step = len(train_loader)

# size of the input is 3*32*32

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2), 
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(192,384, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(384,256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.features(x);
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

model = AlexNet(num_classes).to(device)
model.load_state_dict(torch.load(modelSavedPath))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
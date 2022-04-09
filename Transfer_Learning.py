# Transfer Learning Script: 

# Import packages:
import torch
from torch import nn 
from torch.optim import SGD 
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import numpy as np

# Hyperparameters: 
number_epochs = 5
batch_size = 5
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

# Transforms: 
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}


# Initialise datasets and dataloaders:
# Images obtained from a folder. 
data = '<ENTER NAME OF FOLDER FOR DATA>'
data_path = f'./Transfer_Learning_Data/{data}/'
phases = ['Training', 'Testing']


datasets_dict = {phase : datasets.ImageFolder(f'{data_path}{phase}',
                         data_transforms[phase])
                         for phase in phases}

dataloaders = {phase : DataLoader(datasets_dict[phase], shuffle=True, 
                                  batch_size=batch_size)
                                  for phase in phases}
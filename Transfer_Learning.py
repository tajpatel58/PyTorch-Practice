# Transfer Learning Script: 

# Import packages:
import torch
from torch import nn 
from torch.optim import SGD 
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import numpy as np
import copy 
import os

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

#Defining a function to train our model: 
def train_model(model, optimizer, num_epochs, lr_scheduler, loss_func):
    """
    A function which applies the training loop for us. 


    Parameters:
    model : (torch.nn.Module) - Initial Deep Learning Model
    optimizer : (optimizer) - Iterative process used to optimize the loss function. 
    num_epochs : (int) - Number of epochs to be used in training. 
    lr_scheduler : Learning Rate optimiser. 
    loss_func : Loss function used for model. 
    
    Returns:
    model : (torch.nn.Module) - Optimized Deep Learning Model
    """

    best_model = copy.deepcopy(model.state_dict())
    best_testing_acc = 0

    for epoch in range(num_epochs):
        error = 0
        # Save the state of the model
        model_state = copy.deepcopy(model.state_dict())

        for phase in phases:
            if phase == 'Training':
                # Set Model to training mode. (This accounts for dropout layers etc)
                model.train()
            else:
                # Set model to testing/validation mode. 
                model.eval() 

            for i, (data, labels) in enumerate(dataloaders[phase]):
                predictions = model(data)
                
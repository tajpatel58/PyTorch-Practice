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
    best_testing_accuracy = 0

    for epoch in range(num_epochs):
        # Save the state of the model
        model_state = copy.deepcopy(model.state_dict())
        epoch_accuracy = 0
        epoch_loss = 0

        for phase in phases:
            if phase == 'Training':
                # Set Model to training mode. (This accounts for dropout layers etc)
                model.train()
            else:
                # Set model to testing/validation mode. 
                model.eval() 

            for batch_no, (data, labels) in enumerate(dataloaders[phase]):
                
                # If we are training, then we need to track the operations for the 
                # optimiser. 
                with torch.set_grad_enabled(phase == 'Training'):
                    # Feed our batch through our Neural Network
                    feed_forward = model(data)
                    # Obtain the class predicted:
                    # Note: torch.max outputs a tuple, (max, max_args)
                    _, predictions = torch.max(feed_forward, axis=1)
                    error = loss_func(predictions, labels)
                
                # If in the training phase, need to apply the optimisation step:
                if phase == 'Training':
                    error.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if phase == 'Testing':
                    epoch_accuracy += torch.sum(predictions == labels)
                    epoch_loss += error.item()
        
        epoch_accuracy = np.float32(epoch_accuracy) / len(datasets_dict['Testing'])
        
        # After each epoch, store the model with the highest proportion of correct predictions.
        if epoch_accuracy >= best_testing_accuracy:
            best_model = model.state_dict()
            best_testing_accuracy = epoch_accuracy
    
    model.load_state_dict(best_model)
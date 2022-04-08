# Transfer Learning Script: 

# Import packages:
import torch
from torch import nn 
from torch.optim import SGD 
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

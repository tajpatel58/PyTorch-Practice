# PyTorch-Practice

I've currently got minimal experience with PyTorch however given it's wide applications, I wanted to become familiar with implementations and thought I'd document my journey. 

#### PyTorch Notes - Markdown:

- Comments on key parts of PyTorch including: Dynamic computation graphs, Optimising, Datasets, Dataloaders, Softmax & CrossEntropy, Transforms, Activation Functions. 

#### PyTorch-Basics - Notebook:

- Covers creating tensors and basic operations. 
- Implements basic Linear Regression Model on the California Housing Project. 
- Explains what a DCG is and how backpropogation works. 

#### Logistic Regression - Notebook: 

- Fits a Logisitc regression model on the Breast Cancer Dataset.
- Demonstrates the basics of inheritance from nn.Module. 

#### Feed_Forward - Notebook: 

- Fits a Neural Network on the MNIST Dataset. 
- Creating a multi-layered Neural Network using the nn.Module, can view and how to use the Activation Functions and linear models to feed a datapoint through a Neural Net. 
- Shows training data in Batches using: Dataloaders, Cross Entropy Loss, Adam optimiser. 
- How to visualise tensors using Matplotlib. 

#### Convolutional Neural Networks - Notebook: 

- First implementation of a CNN on the Cifar-10 Dataset. 
- Gives a step by step overview of how the model is being trained. 
- Building familiarity with Pooling and Convolutional Filter layers to reduce feature size. 


#### Transfer Learning - Python Script: 

- This script covers how we can use an already trained model such as Resnet18 to extract the features from an image. 
- Shows how to adapt the fully connected layers on the pretrained model for our use case. 
- Built intuition on common transformations on image data. 
- Explains how to save/load model params so we ultimately choose the params that had the highest epoch accuracy (or any other metric we'd like to use)



#### Conclusion: 

I feel I've built a good foundation with PyTorch and think it's time to move onto bigger projects. My first PyTorch project will be: Facebook Marketplace's Recommendation Ranking System. The aim of which will be to recommend products on Facebook based upon a users search history. 

Facebook Project can be found on my Github. :) 
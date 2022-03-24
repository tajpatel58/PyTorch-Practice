## PyTorch Notes:

- PyTorch is a deep learning library, it has many built in functions to the point where for the most part, the hard work is in designing the model itself. 
- This is a collection of notes that should help build some familiarity with PyTorch as well as provides some tips and tricks. 

### Importing PyTorch into Python:

- Import PyTorch with "import torch"

### Tensor Basics:

- Create a tensor with 0's/1's with the "empty/ones" methods, specifying the dimensions. 
- Create a random tensor with the "rand" method, again specifying the dimensions. 
- Use "tensor" to create your own custom tensor.
- Underscores after the method name can be used to do operations inplace. 
- Can splice just like we do with numpy arrays.
- .item() can be used to retrieve value from a scalar from a tesnor. 
- "view" method can be used to reshape tensors, specify dimensions. Note: useful to set one of the arguments to be -1 to say that we fill the array based on the other dimension. eg. a tensor is 2x2, then applying .view(4, -1) this tells PyTorch to convert to a 4x1.  
- Cast from tensor -> numpy array with .numpy(), note this method means that both objects share the same memory location. 
- Cast from numpy -> tensor with ".from_numpy()" method. 
- Often when creating tensors, can provide the keyword argument "requires_grad" to be "True". This tells PyTorch we will need the gradient wrt this tensor later. 
- "*" does element wise multiplication.  
- It's good practice to set the data type of a tensor. 

### Gradient Calculation: 

- The purpose of the "requires_grad" argument when creating a tensor is to compute derivates with respect to these variables. 
- Once we set requires_grad = True, whenever we apply an operation to this tensor, the operation will be tracked and the computation to get the derivative of the original tensors variables will be tracked. For example: if x is a tensor, then y=x+2, y is also a tensor. y will be stored with grad_fn = "AddBackward0" an arrow on the Dynamic Computation Graph. 
- The "backward" method can be called to compute intermediate derivates. Consider if z = y * y(element wise), y = x+2. Then z.backward() will compute dz/dy, then computes dy/dx, then applies Chain Rule. 
- These "steps" form a Dynamic Computation Graph which is stored when operations are written. 
- Graph nodes tend to be variables and arrows are mathematical operations. 
- Epochs are a way of defining the iteration of gradient descent we're on. 
- Use the context manager "with torch.no_grad()" to update the model params without adding to the computational graph. 
- After an update step of the model params in Gradient Descent, we should also call: ".grad_zero_()", this is because PyTorch accumulates gradients. 

### Training Pipeline:

- For training and designing a model, we can use the "nn" module from PyTorch: "import torch.nn as nn".
- This module contains a lot of the functions we would need: nn.MSELoss(), nn.optim.SGD([w], lr=learning_rate) for computing the loss and applying gradient descent. 
- The SGD function takes in the params that you'd expect: learning rate and the weights and we can call nn.optim.SGD([w], lr=learning_rate).step() to apply one iteration of Gradient descent.
- We can create instance of models using the "nn" module. eg. for a linear regression use: model = nn.Linear(input_size, output_size). This might seem confusing initially as since when does Linear Regression require input_size and output_size. Well if we consider Linear Regression in a neural network format, then this is essentially a single layered network where input_size = number of features and output_size = 1. (With g(x) = 1 as the activation function). 
- Once we've created a model/network then the method of "feeding forward" through our network is well defined. To feed forward using a model we just pass the design matrix/data points as an argumnet to model. eg: model(X_test).
- Note that when we create an instance of a model like above, the weights are randomly initialised as expected.
- The model params can be retrieved with "model.parameters()".


### Processing Data: 

#### Datasets and Dataloaders:

- Note an Epoch is the number of times the entire dataset will be parsed into the optimizer. 
- If we're dealing with big datasets, then we should parse out data into the optimizer in Batches. 
- This is where the "Datasets" and "Dataloaders" objects are useful. Dataset objects are given as arguments into Dataloaders. Dataloaders are used to used to wrap an iterable around the dataset, that way we can parse out data into batches to our optimizers. 
- To do this, we should create a class which inherits from the Dataset class. Define the __getitem__, __len__, __init__ methods. Note for the __getitem__ method, we should output a tuple with the datapoint and the target. 
- Parse this dataset object into the Dataloader so we can then loop through our data. Dataloaders take in some useful __keyword__ arguments: batch_size, shuffle, num_workers (number of cores used). 

#### Transforms:
- Typically our data will need some form of preprocessing before it's parsed into our model. 
- There are many built in transforms that we can use, a few examples are: rotations, grayscale, resize, casting tpyes. eg(numpy -> tensor)
- We can also use custom transforms: for modularity, we should do this processing in a "transform" class. 
- In our transform class we only need to define the "__call__" magic method containing what the transform should do to each datapoint. 
- Recall, we parse the Dataset object into the Dataloader, all we need to do is ammend the __getitem__ magic method in the Dataset clas so that the datapoint that is returned has been transformed. As we've defined the "__call__" method, this can be done with a call to the class. 
- Note: Parse the transformer instance as a keyword argument into the extended Dataset class, that way we can make use of the call functionality. 
- Don't forget, the call method needs to be passed a "sample" to transform. 
- The Compose object can take in a list of multiple transforms and return a new transform. Which we can use to apply multiple transforms in chronological order. 

### Softmax and Cross Entropy:
- The Softmax function is used to assign probabilities by normalising a vector.
- Given an n-dimensional vector of reals, it maps each by taking the exponent and dividing by the exponent of the sum of the values in the vector. 
- Given a k-class classification problem, we apply the Softmax function to the k-dimensional vector returned from the neural net to give a vector of probabilities where the ith value is the probability of belonging to the ith class. 
- The Cross Entropy function takes in 2 arguments: one is the the output vector of a neural net, the second is the actual class that the data point that was fed through the network corresponds to.
- The CE function applies the softmax function to an ouput of a Neural Net and computes the classification error as: -log(p_i) where p_i is the probability of the datapoint belonging to the class. This makes intuitive sense, if the probability is 1, then we get the error as 0, and the smaller p_i is, the progressively larger the error becomes. In other words if the probability of the datapoint actually belonging to the class is small (according to our model), then the error is large. 


### Activation Functions:
- Activation functions are applied between layers of a Neural Net, some activation functions have particular use cases which I'll outline below.
- Used in Binary Logistic regression to associate probability to classes. 
- TanH - Used in the hidden layers. Values between (-1,1). 
- ReLU = max(0,x), rule of thumb: use ReLU when we have no prior idea of the activation function. 
- Leaky ReLU is like ReLU for x>0, but for x<0, we return a*x where a is some gradient. This function is used to try solve the vanishing gradient problem: In particular recall the Backpropogation step requires the derivative of the activation function. A plot of the ReLU shows it has a 0 derivative for x<0, this means that the values at these nodes in our Neural net will never be updated. A good idea is if weights aren't updating through SGD, then this we can try the Leaky ReLU. 
- Softmax used for multiclass classifiction in the final layer. 
- These activation functions can be found in the "nn" module in PyTorch


### Convoluted Neural Networks: 
- CNNs (Convoluted Neural Networks) are quite similar to Nerual Networks however they differ because of the convolutional layer in the network which entail applying a convolution filter on some layers. 
- The typical architecture of a CNN will also involve a "pooling" layer. 
- So what is a convolution filter? 

![](./Graphics/2022-03-24-21-03-20.png)

- The graphic above shows how a convolution filter is applied to a matrix. 
- Essentially, the convolutional filter is a smaller dimension matrix which maps a matrix to a another matrix using the following process:
    1. Place the filter matrix "over" the original matrix in a "window" like manner. 
    2. Moving the convolution filter across the input matrix, will assign entries in the output matrix by taking an element wise sum. 
    3. For example, if we had a 3x3 input matrix, and a 2x2 convolution filter, the output matrix will be a 2x2. 
- Convolutional layers can be used to reduce the number of features throughout the layers of a neural network. 
- An example would be if we applied a CNN to images which are usually 3D tensors. An RGB picture is made of pixels of different colours,(consider a 5x5 picture), each pixel has a corresponding R(red), G(green), B(blue) value between 0 and 255 (totalling 75 nodes on the input layer). We can think of these 3 colour groups as 3 5x5 matrices. We can apply a different convolutional filter to each of these matrices, say a 3x3 matrix. Doing this will output 3 3x3 matrices, reducing the number of features from 75 to 27. 



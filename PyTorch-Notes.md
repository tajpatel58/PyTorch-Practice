## PyTorch Notes:

- Importing PyTorch with  "import torch"

### Tensor Basics:

- Create a tensor with 0's/1's with the "empty/ones" methods, specifying the dimensions. 
- Create a random tensor with the "rand" method, again specifying the dimensions. 
- Use "tensor" to create your own custom tensor.
- Underscores after the method name can be used to do operations inplace. 
- Can splice just like we do with numpy arrays.
- .item() can be used to retrieve value from a scalar from a tesnor. 
- view method can be used to reshape tensors, specify dimensions. 
- Cast from tensor -> numpy array with .numpy(), note this method means that both objects share the same memory location. 
- Cast from numpy -> tensor with ".from_numpy()" method. 
- Often when creating tensors, can provide the keyword argument "requires_grad" to be "True". This tells PyTorch we will need the gradient of this tensor later. Recall, tensors correspond to mappings.
- "*" does element wise multiplication.  

### Gradient Calculation: 

- The purpose of the "requires_grad" argument when creating a tensor is to compute derivates with respect to these variables. 
- Once we set requires_grad = True, whenever we apply an operation to this tensor, the operation will be tracked and the computation to get the derivative of the original tensors variables will be tracked. For example: if x is a tensor, then y=x+2, y is also a tensor. y will be stored with grad_fn = "AddBackward0" which indicates how to get the derivative of x from y.
- The "backward" method can be called to compute intermediate derivates. Consider if z = y * y(element wise), y = x+2. Then z.backward() will compute dz/dy, then computes dy/dx, then applies Chain Rule. 
- These "steps" form a Dynamic Computation Graph which is stored when operations are written. 
- Graph nodes tend to be variables and arrows are mathematical operations. 


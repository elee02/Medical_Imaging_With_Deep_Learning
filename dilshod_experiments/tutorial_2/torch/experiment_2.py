''' Tensors '''

import torch
import numpy as np
print(torch.Tensor(3,4,2)) # Creates Tensor with given shape

print(torch.Tensor(([[1, 2], [3, 4]]))) # (input list): Creates a tensor from the list elements 

print(torch.zeros(2,4)) # Creates 2D Tensor filled with zeros

print(torch.ones(2,2,2)) # Creates 3D Tensor filled with ones

print(torch.rand(3,2)) # Creates a tensor with random values uniformly sampled between 0 and 1

print(torch.randn(4,2)) # Creates a tensor with random values sampled from a normal distribution with mean 0 and variance 1

print(torch.arange(5)) # Creates a tensor containing the values N,N+1,N+2,...,M

print(torch.arange(2,6,0.5)) # Creates a Tensor: (start, end, step)

x = np.array([0, 1, 2, 3, 4])
y = torch.from_numpy(x) # Converts a numpy array into a PyTorch tensor
print(f'Numpy array: {x}\nConverted to PyTorch tensor: {y}')

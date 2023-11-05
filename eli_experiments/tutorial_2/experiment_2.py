'''Tensor initialization experiments.'''
import torch
import numpy as np

a = torch.Tensor(torch.rand(2, 3)) # 2x3 matrix of random numbers between 0 and 1
print(a)

b = torch.ones(2, 3) # 2x3 matrix of ones
print(b)

c = torch.Tensor(a) # copy of a
print(c)

d = torch.randn(5) # 5x1 matrix of random numbers (normal distribution)
print(d)

e = torch.arange(1,5) # 4x1 matrix of numbers from 1 to 4
print(e)

f = np.array([1, 2, 3, 4, 5])
g = torch.from_numpy(f) # convert numpy array to tensor
print(f'Numpy array: {f}\nConverted to Torch: {g}')
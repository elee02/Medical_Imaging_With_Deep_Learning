''' Getting Shape of Tensor (same as in numpy)'''
import torch

x = torch.Tensor(3,4,2)

shape = x.shape
print("Shape:", shape)

size = x.size()
print("Size:", size)

dim1, dim2, dim3 = x.size()
print("Size:", dim1, dim2, dim3)
''' Getting Shape of Tensor '''
import torch

x = torch.Tensor(3,4,2)

shape = x.shape
print("Shape:", shape)

size = x.size()
print("Size:", size)

dim1, dim2, dim3 = x.size()
print("dim1, dim2, dim3:", dim1, dim2, dim3)

print(x.size(0)) # size of the dimension at the index 0
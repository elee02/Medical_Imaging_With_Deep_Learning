'''Getting the size/shape of a tensor'''
import torch

x = torch.rand([5,11,3,2])
print(x.shape) # overall shape

print(x.size(2)) # size of the dimension at the index 2 (third dimesion)
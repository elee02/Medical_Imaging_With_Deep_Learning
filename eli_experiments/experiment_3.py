'''Getting the size/shape of a tensor'''
import torch

x = torch.rand([5,11,3,2])
print(x.shape)

print(x.size(2))
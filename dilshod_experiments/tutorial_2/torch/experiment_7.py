''' Gradients '''
import torch

x = torch.ones(4) # tensors by default does not require gradients
print(x.requires_grad) 

x.requires_grad_(True) # underscore indicating that this is a in-place operation
print(x.requires_grad)

y = torch.ones(2,3, requires_grad=True) # pass the argument `requires_grad=True/False`
print(x.requires_grad)


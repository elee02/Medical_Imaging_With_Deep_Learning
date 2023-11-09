''' GPU/CPU Device '''
import torch

# Good practice to define 'device' object
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device: ", device)

# By default all tensors created are stored in CPU
x = torch.ones(3,3)
print("Tensor is stored: ", x.device)

# Pushing tensor to GPU

# .to() (GPU/CPU)
x = x.to(device)
print("Tensor is stored: ", x.device)
print()

# .cuda() (GPU)
y = torch.ones(2,2)
print("Tensor is stored: ", y.device)
y = y.cuda(device) # device must be cuda 
print("Tensor is stored: ", y.device)



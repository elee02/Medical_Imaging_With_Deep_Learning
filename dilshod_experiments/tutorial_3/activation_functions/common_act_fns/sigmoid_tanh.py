import torch
import torch.nn as nn

# Base class derived from nn.Module
class ActivationFunction(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        # 'config' dictionary to store adjustable parameters for some activation functions
        self.config = {"name": self.name}

# Implementing Sigmoid activation function
# torch.sigmoid in PyTorch
class Sigmoid(ActivationFunction):

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

# Implementing Tanh activation function
# torch.tanh in PyTorch 
class Tanh(ActivationFunction):
    
    def forward(self, x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    

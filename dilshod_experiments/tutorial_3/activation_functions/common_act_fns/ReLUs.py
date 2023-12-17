import torch
import torch.nn as nn

# Base class derived from nn.Module
class ActivationFunction(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        # 'config' dictionary to store adjustable parameters for some activation functions
        self.config = {"name": self.name}

# ReLU activation function
class ReLU(ActivationFunction):
    
    def forward(self, x):
        return x * (x > 0).float()


# LeakyReLU activation function
class LeakyReLU(ActivationFunction):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.config["alpha"] = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.config["alpha"] * x)
    
# ELU activation function
class ELU(ActivationFunction):
    
    def forward(self, x):
        return torch.where(x > 0, x, torch.exp(x)-1)
    
# Swish activation function
class Swish(ActivationFunction):
    
    def forward(self, x):
        return x * torch.sigmoid(x)
    
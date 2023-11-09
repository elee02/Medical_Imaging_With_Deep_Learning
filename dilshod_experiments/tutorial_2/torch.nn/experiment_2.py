''' nn.Module '''
import torch.nn as nn

"""
In PyTorch, a neural network is build up out of modules.
Modules can contain other modules, 
and a neural network is considered to be a module itself as well. 
The basic template of a module is as follows:
"""
class MyModule(nn.Module):

    def __init__(self):
        super().__init__()
        # Some init for my module

    def forward(self, x):
        # Function for performing the calculation of the module.
        pass

# The backward calculation is done automatically, but could be overwritten.
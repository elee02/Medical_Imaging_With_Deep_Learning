'''Setting the seed for reproducibility '''
import torch
torch.manual_seed(0)
print(torch.rand()) # Run twice to see the same random numbers
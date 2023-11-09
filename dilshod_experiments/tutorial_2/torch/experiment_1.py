''' Setting seed to get reproducible random numbers '''
import torch
torch.manual_seed(42)  # Setting the seed
print(torch.rand(2,3)) # Run twice to get the same random numbers
''' GPU seed '''
import torch

# the seed between CPU and GPU is not synchronized

# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(42) # setting seed both GPU/CPU
    torch.cuda.manual_seed_all(42) # setting seed all GPUs


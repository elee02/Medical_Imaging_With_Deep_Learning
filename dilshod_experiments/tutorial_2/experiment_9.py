''' GPU Support '''
import torch

gpu_avail = torch.cuda.is_available()
print("GPU available: ", gpu_avail)
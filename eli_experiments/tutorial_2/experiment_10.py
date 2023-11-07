'''Pushing the tensors to GPU'''
import torch as t
# creating a device object to identify if
# your device posesses any GPU (CUDA)
device = t.device('gpu') if t.cuda.is_available() else t.device('cpu')
print('Device:', device)
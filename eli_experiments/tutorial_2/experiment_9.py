'''Checking if you have a GPU (CUDA)'''
import torch as t 

avail = t.cuda.is_available()
print("Is GPU availabe?", avail)
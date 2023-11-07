''' Indexing '''
import torch

x = torch.arange(20).view(4,5)

print('x: ', x)
print()
print('First raw: ', x[0])
print('Last raw: ', x[-1])
print()
print('First column: ', x[:, 0])
print('Second-last column: ', x[:, -2])
print()
print('Middle column: ', x[:, 2])
print('Middle two raws: ', x[1:3])

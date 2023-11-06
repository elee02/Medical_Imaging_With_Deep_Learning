''' Operations '''
import torch

print('\nPERMUTATION\n')
x = torch.arange(12)
print('x\n', x)
x = x.view(2,3,2) # reshaping tensor
print(f'x (reshaped) \n {x.size()}\n {x}')
x = x.permute(1,0,2) # permuting tensor (sequence of dimension indices)
print(f'x (permuted (1,0,2)) \n {x.size()}\n {x}')

print('\nMULTIPLICATION (mm - only 2D)\n')
x1 = torch.arange(12).view(3,4) # We can also stack multiple operations in a single line
x2 = torch.arange(2,10).view(4,2)
print('x1 ', x1)
print('x2 ', x2)
print('multiplication:\n', torch.mm(x1, x2))

print('\nMULTIPLICATION (matmul - implicit broadcasting)\n')
m1 = torch.arange(12).view(2,2,3)
m2 = torch.arange(2,14).view(3,4)
print('m1 ', m1)
print('m2 ', m2)
print('multiplication:\n', torch.matmul(m1, m2))

print('\nMULTIPLICATION (matmul - explicit broadcasting)\n')
y1 = torch.arange(4).view(1,4)
y2 = torch.arange(16).view(4,4)
print('y1 ', y1)
print('y2 ', y2)
# Explicitly broadcast y2 to match the shape of y1
y1_b, y2_b = torch.broadcast_tensors(y1, y2)
print('y1_b ', y1_b)
print('y2_b ', y2_b)
print('multiplication:\n', torch.matmul(y1_b, y2_b))


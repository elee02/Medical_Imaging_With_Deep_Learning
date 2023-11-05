''' Operations '''
import torch

print('ADDITION')
x1 = torch.rand(2,2)
x2 = torch.rand(2,2)

y = x1 + x2 # addition

print('x1:\n', x1)
print('x2:\n', x2)
print('y:\n', y)

print('INLINE ADDITION')
print('x1 (before):\n', x1)
print('x2 (before):\n', x2)

x2.add_(x1) # inline addition

print('x1 (after):\n', x1)
print('x2 (after):\n', x2)

print('CHANGING TENSOR SHAPE')
x = torch.arange(6)
print('x:\n',x)

x = x.view(3,2) # reshaping to 3x2 Tensor

print('x (reshaped):\n', x)
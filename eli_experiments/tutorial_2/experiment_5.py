'''Reshaping and transposing the tensors'''
import torch as t

print("Matrix of 4x6:")
a = t.zeros(4,6)
print(a)

# reshape the tensor with .view method
print()
print("Reshaped matirx of 3x8:")
a = a.view(3,8)
print(a)

# transpose with .permute() method
print()
print("Transposed matrix of 8x3:")
a = a.permute(1, 0) # requires the dimensions that are permuted
print(a)

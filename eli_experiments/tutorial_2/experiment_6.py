'''Multiplication of matrices with broadcasting (.matmul() method)'''
import torch as t

# multiplying 2D arrays
a = t.arange(5,20).view(3, 5)
b = t.randn(3, 5)
print(f'First tensor:\n{a}')
print()
print(f'Second tensor:\n{b}')

print()
c = a * b # element-wise or Hadamard product
print(f'The element-wise product:\n{c}')

print() 
d = t.mm(a, b) # matrix multiplication
print(f"The 2D matrix multiplication:\n{d}")

# in the case of multiplication of different
# shapes, broatcasting is applied
print()
e = t.ones(3, 11, 7) * 5
f = t.ones(4, 1, 7, 2) * 2
print(f"Third tensor:\n{e}\n\n{e.shape}")
print()
print(f"Fourth tensor:\n{f}\n\n{f.shape}")

g = e @ f
print()
print(f"Broatcased multiplication:\n{g}\n")
print(f"The shape of result: {g.shape}")
print()
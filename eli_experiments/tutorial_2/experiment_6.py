'''Multiplication of matrices with broadcasting (.matmul() method)'''
import torch as t

# multiplying 2D arrays
a = t.arange(5,20).view(3, 5)
b = t.randn(3, 5)
print(f'First tensor:\n{a}')
print()
print(f'Second tensor:\n{b}')

print()
c = a * b
print(f'The multiplication:\n{c}')

# in the case of multiplication of different
# shapes, broatcasting is applied
print()
d = t.ones(3, 11, 7) * 5
e = t.ones(4, 1, 7, 2) * 2
print(f"Third tensor:\n{d}\n\n{d.shape}")
print()
print(f"Fourth tensor:\n{e}\n\n{e.shape}")

f = d @ e
print()
print(f"Broatcased multiplication:\n{f}\n")
print(f"The shape of result: {f.shape}")
print()
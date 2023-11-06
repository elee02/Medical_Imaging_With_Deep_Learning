'''Multiplication without broadcasting (.mm method)'''
import torch as t

a = t.rand(2,5)
b = t.rand(5,4)

c = t.mm(a, b) # a and b MUST be only 2D arrays
print(c)
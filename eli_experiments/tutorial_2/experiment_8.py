'''Creating a tensor requiring a gredient'''
import torch as t

x = t.arange(3.0,requires_grad=True)
a = x + 2
b = a ** 2
c = b + 3
y = c.mean()
y.backward()

print(f"Input Tensor:\n{a}\n\nOutput Tensor:\n{b}\n\nGradient: {x.grad}")
''' Dynamic Computation Graph and Backpropagation '''
import torch

print("y = 1/|x| * Σ [(x_i + 2)^2 + 3]") # function

x = torch.arange(3.0, requires_grad=True) # only float tensors can have gradients

# computation graph step by step
a = x + 2
print(a)
b = a ** 2
print(b)
c = b + 3
print(c)
y = c.mean()
print("Y", y)

# perform backpropagation on the computation graph 
# by calling the function `backward()` on the last output
y.backward()

# x.grad will now contain the gradient dy/dx
# this gradient indicates how a change in x
# will affect output y given the current input x=[0,1,2]
print(x.grad)
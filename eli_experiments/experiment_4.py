'''Simple Operations'''
import torch as t
# using addition operator (+) - copy
a = t.arange(0,5)
b = t.arange(5,10)

c = a + b
print("Using addition operator (+) - copy")
print(f'a: {a}\
\nb: {b}\
\na + b: {c}')

# using .add_() method - in-place
print()
print("Using .add_() method - in-place")
print(f'Before:\
\na: {a}\
\nb: {b}\
\nAfter:\
\na: {a}\
\nb.add_(b): {b.add_(a)}\
\nb: {b}')


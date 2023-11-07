'''Ensuring reproducibility with Intel GPUs'''
import torch as t

if t.cuda.is_available():
    t.cuda.manual_seed(42)
    t.cuda.manual_seed_all(42)

# Prevent stochastic efficeincy algorithms
# for reproducibility

t.backends.cudnn.deterministic = True # Some cuDNN algorithms use non-deterministic operations. Setting this flag to True forces cuDNN to use only deterministic ones.
t.backends.cudnn.benchmark = False # This line tells cuDNN not to optimize the compute-intensive operations in your neural network. By default, cuDNN tries to find the best algorithm for your GPU and the size of your input data, which can lead to some variability in performance. Setting this flag to False disables this behavior.

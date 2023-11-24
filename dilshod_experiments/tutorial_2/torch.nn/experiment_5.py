""" The Data Loader Class """

# The class torch.utils.data.DataLoader represents a Python iterable over a dataset

# The data loader communicates with the dataset using the function __getitem__,
# and stacks its outputs as tensors over the first dimension to form a batch


# Data Loader input arguments: 

# full list https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

# batch_size: Number of samples to stack per batch

# shuffle: If True, the data is returned in a random order
# This is important during training for introducing stochasticity.

# num_workers: Number of subprocesses to use for data loading
# The default, 0, means that the data will be loaded in the main process

# pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
# This can save some time for large data points on GPUs

# drop_last: If True, the last batch is dropped in case it is smaller than the specified batch size
# This occurs when the dataset size is not a multiple of the batch size
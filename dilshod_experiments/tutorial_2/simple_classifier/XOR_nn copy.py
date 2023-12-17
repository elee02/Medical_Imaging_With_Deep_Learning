import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

class SimpleClassifier(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        # call initializer(constructor) of the nn.Module class
        super().__init__()

        # create the first linear layer of the network
        # applied transformation: y = xA^T + b
        # x - input, A - weight matrix, b - bias, y - output
        self.linear1 = nn.Linear(num_inputs, num_hidden)

        # create the second linear layer of the network
        # similar linear transformation is applied 
        self.linear2 = nn.Linear(num_hidden, num_outputs)

        # implement the activation function using nn.Tanh
        self.act_fn = nn.Tanh()

        
    def forward(self, x):
        # Perform the calculation of the model to determine the prediction

        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)

        return x


class XORDataset(data.Dataset):

    def __init__(self, size, std=0.1):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()


    def generate_continuous_xor(self):
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)

        # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label

def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:

            ## Step 1 - Move input data to device
            # data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)

            ## Step 2 - Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3 - Calculate the loss using the module loss_module
            loss = loss_module(preds, data_labels.float())

            ## Step 4 - Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5 - Update parameters
            optimizer.step()


def eval_model(model, data_loader):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.

    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:

            ## TODO: Step 1 - Move data to device
            # data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)

            ## TODO: Step 2 - Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1

            ## TODO: Step 3 - Determine binary predictions of the model
            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1

            ## TODO: Step 4 - Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print("Accuracy of the model: %4.2f%%" % (100.0*acc))


# MAIN
model = SimpleClassifier(num_inputs = 2, num_hidden = 4, num_outputs = 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

loss_module = nn.BCEWithLogitsLoss()

train_dataset = XORDataset(size=1000)

train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# Push model to device. Has to be only done once
# model.to(device)

train_model(model, optimizer, train_data_loader, loss_module)


test_dataset = XORDataset(size=500)

test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)

eval_model(model, test_data_loader)

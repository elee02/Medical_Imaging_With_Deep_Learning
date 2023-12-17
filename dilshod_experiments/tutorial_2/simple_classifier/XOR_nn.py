import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

class SimpleClassifier(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_outputs)
        self.act_fn = nn.Tanh()

        
    def forward(self, x):
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
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)

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
    model.train()

    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:

            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            loss = loss_module(preds, data_labels.float())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


def eval_model(model, data_loader):
    model.eval()
    true_preds, num_preds = 0., 0.

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:

            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds)

            pred_labels = (preds >= 0.5).long() 

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

# model.to(device)

# Training
train_model(model, optimizer, train_data_loader, loss_module)


# Testing
test_dataset = XORDataset(size=500)

test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)

eval_model(model, test_data_loader)
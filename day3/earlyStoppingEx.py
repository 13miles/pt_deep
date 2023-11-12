import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(111)

import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def create_datasets(batch_size):
    train_data = dset.MNIST(root='MNIST_data/',
                             train=True,
                             download=True,
                             transform=transforms.ToTensor())

    test_data = dset.MNIST(root='MNIST_data/',
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data,  batch_size=batch_size,  sampler=train_sampler)
    valid_loader = DataLoader(train_data,  batch_size=batch_size,  sampler=valid_sampler)
    test_loader = DataLoader(test_data,batch_size=batch_size)

    return train_loader, test_loader, valid_loader


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        y = self.fc3(x)
        return y

model = Net()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'validation loss: ({self.val_loss_min:.6f}) -> ({val_loss:.6f})  saving model!!!!!')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'earlyStopping counter:{self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0


def train_model(model, patience, n_epochs):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs+1):
        model.train()
        for batch, (x_data, target) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            hypothesis = model(x_data)
            loss = loss_func(hypothesis, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        for x_data, target in valid_loader:
            output = model(x_data)
            loss = loss_func(output, target)
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(n_epochs))

        print(f'[{epoch:>{epoch_len}} / {n_epochs:>{epoch_len}}] '+
              f'train_loss:{train_loss:.5f} valid_loss{valid_loss:.5f}')
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print('early stopping!!!!')
            break

    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses

batch_size = 256
n_epochs = 100

train_loader, test_loader, valid_loader = create_datasets(batch_size)
patience = 20
model, train_loss, valid_loss = train_model(model, patience, n_epochs)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')

minposs = valid_loss.index(min(valid_loss))+1
plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5)
plt.xlim(0, len(train_loss)+1)
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
fig.savefig('loss_plot.png', bbox_inches='tight')
plt.show()

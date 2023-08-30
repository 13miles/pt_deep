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

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(train_data, batch_size=batch_size)

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
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        y = self.fc3(x)
        return y

model = Net()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())



class EarlyStopping:
    def __init__(selfself, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf  # 초기값은 무한대
        self.delta = delta
        self.path = path

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'validation loss: ({self.val_loss_min:.6f}) -> ({val_loss:.6f}) saving model!!!')
        torch.save(model.state_dic(), self.path)
        self.val_loss_min = val_loss

    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.count += 1
            print(f'earlyStopping counter / patience :{self.counter / {self.patience}')
            if self.count >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
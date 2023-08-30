import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(111)

train_loader = DataLoader(
    dset.MNIST(root='MNIST_data/',
                         train=True,
                         download=True,
                         transform=transforms.ToTensor()),
    batch_size=100,
    shuffle=True
)

test_loader = DataLoader(
    dset.MNIST(root='MNIST_data/',
                         train=False,
                         download=True,
                         transform=transforms.ToTensor()),
    batch_size=100,
    shuffle=True
)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, drop_p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,10)

        self.dropout_p = drop_p

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout_p)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training, p=self.dropout_p)
        y = self.fc3(x)
        return y

model = Net(drop_p=0.2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        hypothesis = model(data)
        loss = F.cross_entropy(hypothesis, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            hypothesis = model(data)
            test_loss += F.cross_entropy(hypothesis, target).item()
            pred = hypothesis.max(dim=1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for epoch in range(1, 21):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print(f'epoch:{epoch} loss:{test_loss:.4f} accuracy:{test_accuracy:.3f}%')
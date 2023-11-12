import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_epochs = 20
batch_size = 100
learning_rate = 0.0002

mnist_train = dset.MNIST(root='MNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

class AutoEncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 20)
        self.decoder = nn.Linear(20, 784)

    def forward(self, x):
        x = x.view(batch_size, -1)
        eoutput = self.encoder(x)
        y = self.decoder(eoutput).view(batch_size, 1, 28, 28)
        return y

AEModel = AutoEncoderNet()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(AEModel.parameters(), lr=learning_rate)

for i in range(train_epochs):
    for x_data, y_data in data_loader:
        optimizer.zero_grad()
        hypothesis = AEModel(x_data)
        loss = loss_func(hypothesis, x_data)
        loss.backward()
        optimizer.step()
    print('loss:', loss.item())

out_img = torch.squeeze(hypothesis.data)

import matplotlib.pyplot as plt
for i in range(3):
    plt.imshow(torch.squeeze(x_data[i]).numpy(), cmap='gray')
    plt.figure()
    plt.imshow(out_img[i].numpy(), cmap='gray')
    plt.show()
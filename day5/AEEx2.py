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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # 64 * 14 * 14
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 128 * 7 * 7
            nn.Conv2d(128, 256, 3, padding=1),  # 256 * 7 * 7
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        y = x.view(batch_size, -1)
        return y

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), #128 * 14 * 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1), #64 * 14 * 14
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, 1, 1), #16 * 14 * 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1), #1 * 28 * 28
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(batch_size, 256, 7, 7)
        x = self.layer1(x)
        y = self.layer2(x)
        return y

encoder = Encoder()
decoder = Decoder()
loss_func = nn.MSELoss()

parameter = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(parameter, lr=learning_rate)

for i in range(train_epochs):
    for j, (x_data, _) in enumerate(data_loader):
        optimizer.zero_grad()
        eoutput = encoder(x_data)
        hypothesis = decoder(eoutput)

        loss = loss_func(hypothesis, x_data)
        loss.backward()
        optimizer.step()

        if j % 100 == 0:
            print('{}/{} loss:{:.4f}'.format(i+1, j+1, loss.item()))

import matplotlib.pyplot as plt
out_img = torch.square(hypothesis.data)
for i in range(3):
    plt.subplot(121)
    plt.imshow(torch.squeeze(x_data[i]).numpyt(), cmap='gray')
    plt.subplot(122)
    plt.imshow(out_img[i].numpy(), cmap='gray')
    plt.show()
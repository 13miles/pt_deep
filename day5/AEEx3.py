import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset

import matplotlib.pyplot as plt
import numpy as np

total_epochs = 20
batch_size = 64

trainset = dset.FashionMNIST(root='FashionMNIST_data/',
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)

train_loader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True
)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
loss_func = nn.MSELoss()

def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img

def train(autoencoder, train_loader):
    autoencoder.train()
    avg_loss = 0
    for x, _ in train_loader:
        x_data = add_noise(x)
        x_data = x_data.view(-1, 784)
        y_data = x.view(-1, 784)

        encoded, decoded = autoencoder(x_data)

        loss = loss_func(decoded, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    return avg_loss / len(train_loader)

for epoch in range(total_epochs):
    loss = train(autoencoder, train_loader)
    print(f'epoch:{epoch+1} loss:{loss:.4f}')

testset = dset.FashionMNIST('FashionMNIST_data/',
                            train=False,
                            download=True,
                            transform=transforms.ToTensor)

sample_data = testset.data[0].view(-1, 28*28)
sample_data = sample_data.type(torch.FloatTensor)/255.

original_x = sample_data[0]
nosiy_x = add_noise(original_x)
_, recovered_x = autoencoder(nosiy_x)

f, a = plt.subplots(1,3, figsize=(15,15))

original_img = np.reshape(original_x.data.numpy(), (28, 28))
noisy_img = np.reshape(nosiy_x.data.numpy(), (28, 28))
recovered_img = np.reshape(recovered_x.data.numpy(), (28, 28))

a[0].set_title('Original')
a[0].imshow(original_img, cmap='gray')

a[1].set_title('Noisy')
a[1].imshow(noisy_img, cmap='gray')

a[2].set_title('Recovered')
a[2].imshow(recovered_img, cmap='gray')

plt.show()
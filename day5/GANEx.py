import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset

import matplotlib.pyplot as plt
import numpy as np

total_epochs = 30
batch_size = 100

trainset = dset.FashionMNIST(root='FashionMNIST_data/',
                             train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))
                             ]),
                             download=True)

train_loader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True
)

G = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 784),
    nn.ReLU(),
    nn.Tanh()
)


D = nn.Sequential(
    nn.Linear(784, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 128),
    nn.LeakyReLU(0.2),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

loss_func = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

for epoch in range(total_epochs):
    for image, _ in train_loader:
        image = image.view(batch_size, -1)
        real_label = torch.ones(batch_size, 1)
        fake_label = torch.zeros(batch_size, 1)

        outputs = D(image)
        d_loss_real = loss_func(outputs, real_label)

        z = torch.randn(batch_size, 64)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = loss_func(outputs, fake_label)

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = loss_func(outputs, real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f'epoch:{epoch+1}, d_loss:{d_loss.item():.4f} g_loss:{g_loss.item():.4f}')

z = torch.randn(batch_size, 64)
fake_images = G(z)

import numpy as np

for i in range(3):
    fake_images_img = np.reshape(fake_images.data.numpy()[i], (28, 28))
    plt.imshow(fake_images_img, cmap='gray')
    plt.show()
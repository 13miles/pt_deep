import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_wine

wine = load_wine()
#print(wine)
print(wine.keys())
print()

import pandas as pd
df = pd.DataFrame(wine.data, columns=wine.feature_names)
#print(df)
#print(wine.DESCR)

wine_data = wine.data[0:130]
wine_target = wine.target[0:130]
print(wine_data.shape)
print(wine_target)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(wine_data, wine_target, test_size=0.2, random_state=48)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()

x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()

train = TensorDataset(x_train, y_train)
train_loader = DataLoader(train, batch_size=16, shuffle=True)

class CNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 96)
        self.fc2 = nn.Linear(96, 96)
        self.fc3 = nn.Linear(96, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)
        self

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        y = self.fc6(x)
        return y

model = CNet()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    total_loss = 0

    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    if epoch % 10 == 0:
        print('epoch:{} total loss:{:.4f}'.format(epoch+1, total_loss))


prediction = torch.max(model(x_test), dim=1)[1]
accuracy = (prediction == y_test).float().mean()
print('prediction:\n{}\naccuracy:{:.4f}'.format(prediction, accuracy.item()))


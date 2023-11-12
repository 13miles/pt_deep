from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

bos = load_boston()
df = pd.DataFrame(bos.data, columns=bos.feature_names)
df['price'] = bos.target
# print(df.head())


from sklearn.preprocessing import MinMaxScaler

x = df.drop('price', axis=1).to_numpy()
y = df['price'].to_numpy().reshape(-1, 1)

print(x.shape)
print(y.shape)
print()

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x[1])

scaler.fit(y)
y = scaler.transform(y)
print(y)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=48)
traindata = TensorData(x_train, y_train)
train_loader = DataLoader(traindata, batch_size=32, shuffle=True)

testdata = TensorData(x_test, y_test)
test_loader = DataLoader(testdata, batch_size=32, shuffle=True)


class Regress(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        y = F.relu(self.fc3(x))
        return y
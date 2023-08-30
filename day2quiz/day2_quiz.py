import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd

data = pd.read_csv('diabetes.csv')

x_data = data.iloc[:, :-1].values
y_data = data.iloc[:, -1].values

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data).view(-1, 1) #차원 변경

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = self.hidden1(x)
        x = torch.sigmoid(x)
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        return torch.sigmoid(x)

model = LogisticRegression()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    hypothesis = model(x_train)
    loss = F.binary_cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prediction = hypothesis > torch.FloatTensor([0.5])
        correction_prediction = prediction.float() == y_train
        accuracy = correction_prediction.sum().item() / len(correction_prediction)
        print('epoch:{} loss:{:.4f} accuracy:{:2.2f}%'.format(
            epoch+1, loss.item(), accuracy * 100
        ))
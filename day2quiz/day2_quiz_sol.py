from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
import numpy as np
import torch

class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 50)
        self.l2 = nn.Linear(50, 10)
        self.l3 = nn.Linear(10, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))

        return y_pred


model = Model()

cost_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    for i, data in enumerate(train_loader, 0):
        x_train, y_target = data

        hypothesis = model(x_train)
        loss = cost_func(hypothesis, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        prediction = hypothesis > torch.FloatTensor([0.5])
        correction_prediction = prediction.float() == y_target
        accuracy = correction_prediction.sum().item() / len(correction_prediction)
        print('epoch:{} loss:{:.4f} accuracy:{:2.2f}%'.format(epoch, loss.item(), accuracy * 100))

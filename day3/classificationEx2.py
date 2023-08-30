import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

x_train = [[1,2,1,1],
           [2,1,3,2],
           [3,1,3,2],
           [4,1,5,5],
           [1,7,5,5],
           [1,2,5,6],
           [1,6,6,6],
           [1,7,7,7]]

y_train = [2,2,2,1,1,1,0,0]

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

class SoftmaxClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,3)

    def forward(self, x):
        return self.fc1(x)

model = SoftmaxClass()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    hypothesis = model(x_train)
    loss = F.cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch:{} loss:{:.4f}'.format(epoch+1, loss.item()))

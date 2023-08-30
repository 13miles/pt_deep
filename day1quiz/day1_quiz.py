import torch
import torch.nn as nn

x_train = torch.FloatTensor([[73, 80, 75, 65],
                             [93, 88, 93, 88],
                             [89, 91, 90, 76],
                             [96, 98, 100, 99],
                             [73, 65, 70, 100],
                             [84, 98, 90, 100]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142],[188]])

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

for data in dataloader:
    print(data, end='\n\n')

model = nn.Linear(4,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

import torch.nn.functional as F

for epoch in range(20):
    for batch_idx, data in enumerate(dataloader):
        batch_x, batch_y = data
        hypothesis = model(batch_x)
        cost = F.mse_loss(hypothesis, batch_y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(f'{epoch+1}/{batch_idx+1} cost {cost.item():.4f}')
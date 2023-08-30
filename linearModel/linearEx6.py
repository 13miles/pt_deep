import torch
import torch.nn as nn

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 92],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 64, 70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2)

for data in dataloader:
    print(data, end='\n\n')

model = nn.Linear(3,1)
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
        print(f'{epoch+1}/{batch_idx+1} cost{cost.item():.4f}')
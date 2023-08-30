import torch
import torch.nn.init as init
import torch.optim as optim

x = init.uniform_(torch.Tensor(1000, 1), -10, 10)
value = init.normal_(torch.Tensor(1000, 1), std=0.2)
y_target = 2 * x + 3 + value

print(x.size())
print(y_target.size())

import torch.nn as nn

model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters() , lr=0.01)
cost_func = nn.MSELoss()

for epoch in range(100):
    hypothesis = model(x)
    cost = cost_func(hypothesis, y_target)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'epoch:{epoch+1} cost:{cost.item():.4f}')
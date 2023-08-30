import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777)

x = torch.FloatTensor([[0,0], [0,1], [1,0], [1,1]])
y = torch.FloatTensor([[0], [1], [1], [0]])

# model = nn.Sequential(
#     nn.Linear(2, 2, bias=True),
#     nn.Sigmoid(),
#     nn.Linear(2, 1),
#     nn.Sigmoid()
# )

model = nn.Sequential(
    nn.Linear(2, 10, bias=True),   # 첫 번째 은닉층, 10개의 노드
    nn.Sigmoid(),
    nn.Linear(10, 10),             # 두 번째 은닉층, 10개의 노드
    nn.Sigmoid(),
    nn.Linear(10, 10),             # 세 번째 은닉층, 10개의 노드
    nn.Sigmoid(),
    nn.Linear(10, 1),              # 출력층
    nn.Sigmoid()
)

loss_func = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1)

for epoch in range(10000):
    hypothesis = model(x)
    loss = loss_func(hypothesis, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch:{epoch+1} loss:{loss.item():.4f}')

with torch.no_grad():
    hypothesis = model(x)
    prediction = (hypothesis > 0.5).float()
    accuracy = (prediction == y).float().mean()
    print('hypothesis:\n{}\nprediction:\n{}\ntarget\n{}\naccuracy:{:.4f}'.format(
        hypothesis.numpy(), prediction.numpy(), y.numpy(), accuracy.item()
    ))


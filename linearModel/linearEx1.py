import torch

w = torch.tensor(2., requires_grad=True)
y = 2 * w
y.backward()
print('w로 미분한 값', w.grad)

w = torch.tensor(2., requires_grad=True)
for epoch in range(20):
    y = 7 * w
    y.backward()
    print('w로 미분한 값: ', w.grad)
    w.grad.zero_()
    #optimizer.zero_grad()
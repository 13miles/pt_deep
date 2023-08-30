import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, input_size, outputsize):
        super().__init__()

        self.w = nn.Parameter(torch.FloatTensor(torch.randn(input_size, outputsize)), requires_grad=True)
        self.b = nn.Parameter(torch.FloatTensor(torch.randn(outputsize)), requires_grad=True)

    def forward(self, x):
        y = torch.mm(x, self.w) + self.b
        return y

x = torch.FloatTensor(torch.randn(16, 10))

CModel = CustomLinear(10, 5)
y = CModel.forward(x)
print(y)
print()

y2 = CModel(x)
print(y2)

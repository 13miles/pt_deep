import torch.nn as nn
import torch

class CustomerLinear2(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.linear(x)
        return y

x = torch.FloatTensor(torch.randn(16, 10))
CModel = CustomerLinear2(10, 5)
y = CModel(x)
print(y)

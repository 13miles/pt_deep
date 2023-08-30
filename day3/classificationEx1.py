import torch
import torch.nn.functional as F

torch.manual_seed(777)

# x = torch.FloatTensor([[1,2,3], [4,5,6]])
# hypothesis = F.softmax(x, dim=1)
# print(hypothesis)

x = torch.randn(3, 5, requires_grad=True) #weight sum
hypothesis = F.softmax(x, dim=1)
print(hypothesis)
print()

y = torch.randint(5, (3,)).long()
print(y)
print()

y_one_hot = torch.zeros_like(hypothesis)
print(y_one_hot)
print()

y_one_hot = y_one_hot.scatter(1, y.unsqueeze(dim=1), 1)
print(y_one_hot)
print()

print(-(y_one_hot * torch.log(F.softmax(x, dim=1))).sum(dim=1))
print(-(y_one_hot * torch.log(F.softmax(x, dim=1))).sum(dim=1).mean())
print(-(y_one_hot * torch.log_softmax(x, dim=1)).sum(dim=1).mean())
print(F.cross_entropy(x, y))
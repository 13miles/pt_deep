import torch

t1 = torch.tensor([1,2,3])
t2 = torch.tensor([5,6,7])

t3 = t1 + 30
print(t3)
print(t2 ** 2)
print()

t4 = t1 + t2
print(t4)
print()

t5 = torch.tensor([[10,20,30], [40,50,60]])
print(t5)
print()
print(t5 + t1)
print()

t6 = torch.linspace(0,3,10)
print(t6)
print()
print(torch.exp(t6))
print(torch.log(t6))
print()

t7 = torch.tensor([[2,4,6], [1,3,9]])
print(t7)
print()
print(torch.max(t7))
print()
print(torch.max(t7, dim=1))
print()

print(torch.max(t7, dim=1)[0])
print(torch.max(t7, dim=1)[1])

import torch

t1 = torch.tensor([1,2,3,4,5,6]).view(3,2)
t2 = torch.tensor([7,8,9,10,11,12]).view(2,3)

t3 = torch.mm(t1, t2)
print(t3)
print()

print(torch.matmul(t1, t2))

t4 = torch.FloatTensor(2,4,3)
t5 = torch.FloatTensor(2,3,5)

print(torch.bmm(t4, t5).size())
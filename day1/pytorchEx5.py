import torch

t1 = torch.tensor([[1,2,3], [4,5,6]])
print(t1)
print()

print(t1[:, :2])
print()

print(t1 > 4)
print()
print(t1[t1 > 4])

t1[:, 2] = 40
print(t1)
print()

t1[t1 > 4] = 100
print(t1)

t2 = torch.tensor([[1,2,3],[4,5,6]])
t3 = torch.tensor([[7,8,9],[10,11,12]])
print(t2)
print()
print(t3)
print()
t4 = torch.cat([t2,t3], dim=0)
print(t4)
print()

for c in torch.chunk(t4, 4, dim=0):
    print(c, end='\n\n')
print()

for c in torch.chunk(t4, 3, dim=1):
    print(c, end='\n\n')
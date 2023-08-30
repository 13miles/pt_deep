import torch

t1 = torch.FloatTensor([4,5,6,7,8])
print(t1)
print(type(t1))
print(t1.numpy())
print(type(t1.numpy()))

t2 = torch.tensor([[1,2], [3,4,]], dtype=torch.float32)
print(t2)
print(t2.size())
print()

import numpy as np

ndata = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.float32)
t3 = torch.from_numpy(ndata)
print(t3)
import numpy as np
import pandas as pd
#import torch as tc
import torch

ldata = [5,2,8,1,9,3]
tdata = 20,90,50,30
print(ldata)

print(np.__version__)
print(pd.__version__)
print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
exit(0)
import numpy as np

def NOT_AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 가중치와 편향을 조절하여 NOT AND 동작을 구현합니다.

    b = 0.5

    tp = np.sum(w * x) + b

    if tp <= 0:
        return 1
    else:
        return 0

# 테스트
print(NOT_AND(0, 0))  # 출력: 1
print(NOT_AND(0, 1))  # 출력: 0
print(NOT_AND(1, 0))  # 출력: 0
print(NOT_AND(1, 1))  # 출력: 0

import numpy as np

def NOT_AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 가중치와 편향을 조절하여 NOT AND 동작을 구현합니다.

    b = 0.7

    tp = np.sum(w * x) + b

    if tp <= 0:
        return 1
    else:
        return 0

# 테스트
print(NOT_AND(0, 0))  # 출력: 1
print(NOT_AND(0, 1))  # 출력: 1
print(NOT_AND(1, 0))  # 출력: 1
print(NOT_AND(1, 1))  # 출력: 0

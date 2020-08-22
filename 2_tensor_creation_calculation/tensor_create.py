import torch
import numpy as np

arr = [2, 1]
x = torch.tensor(arr, dtype=torch.float32)
print(f'从数组中创建tensor:\n{x}. 数据类型为:{x.dtype}')

arr = [[1, 2], [3, 4]]
x = torch.tensor(arr)
print(f'从二维数组中创建tensor:\n{x}. 数据类型为:{x.dtype}')

x = torch.rand(3, 2)
print(f'随机值 3*2 tensor:\n {x}')

x = torch.zeros(2, 3)
print(f'全0 2*3 tensor:\n{x}')

y = torch.zeros_like(x)
print(f'创建跟x相同大小的全0 tensor:\n{y}')

x = torch.ones(2, 3)
print(f'全1 2*3 tensor:\n{x}')

y = torch.ones_like(x)
print(f'创建跟x相同大小的全1 tensor:\n{y}')

np_arr = np.random.rand(3, 4)
print(f'创建numpy数组:\n{np_arr}')
tensor_arr = torch.from_numpy(np_arr)
print(f'由numpy数组创建tensor:\n{tensor_arr}')

# numpy和pytorch共享相同的内存
# 修改numpy
np_arr[0] = 100
# 修改tensor
tensor_arr[2] = -100
print(f'修改numpy之后numpy数组:\n{np_arr}')
print(f'修改numpy之后pytorch数组:\n{tensor_arr}')

# tensor的大小
x = torch.rand(4, 3)
print(f'x.shape:{x.shape}')
print(f'x.size():{x.size()}')

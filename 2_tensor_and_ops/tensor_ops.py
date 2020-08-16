import torch
import numpy as np

x_1 = [2, 1]
x_2 = [[1, 2], [3, 4]]
y_1 = torch.tensor(x_1)
y_2 = torch.tensor(x_2)

print(y_1, y_2, y_1.size(), y_2.size())

x_1[0] = 100
print(f'x_1={x_1}')
print(f'y_1={y_1}')

x_1 = torch.empty(1) 
x_2 = torch.empty(2, 3)
x_3 = torch.empty(3, 2, 2)

print(f'x_1 value {x_1}')
print(f'x_2 value {x_2}')
print(f'x_3 value {x_3}')

print(f'x_1 size {x_1.size()}')
print(f'x_2 size {x_2.size()}')
print(f'x_3 size {x_3.size()}')

x_1 = torch.rand(3, 2)
print(f'x_1 value {x_1}')
print(f'x_1 size {x_1.size()}')

x_2 = torch.rand(3, 2, 2)
print(f'x_2 value {x_2}')
print(f'x_2 size {x_2.size()}')

x_1 = torch.zeros(2, 3)
print(f'x_1 value {x_1}')
print(f'x_1 size {x_1.size()}')

x_2 = torch.zeros(2, 3, 2)
print(f'x_2 value {x_2}')
print(f'x_2 size {x_2.size()}')

x_1 = torch.ones(2, 3)
print(f'x_1 value {x_1}')
print(f'x_1 size {x_1.size()}')

x_2 = torch.ones(2, 3, 1)
print(f'x_2 value {x_2}')
print(f'x_2 size {x_2.size()}')

x_np = np.ones(3)
x_torch = torch.from_numpy(x_np)
print(f'numpy x:\n{x_np}')
print(f'torch x:\n{x_torch}')

x_np[0] = 100
x_torch[2] = -100
print(f'numpy x after change value:\n{x_np}')
print(f'tensor x after change value:\n{x_torch}')

x_torch = torch.rand(3)
x_np = x_torch.numpy()
print(f'torch x:\n{x_torch}')
print(f'numpy x:\n{x_np}')

x_np[0] = 100
x_torch[2] = -100
print(f'numpy x after change value:\n{x_np}')
print(f'tensor x after change value:\n{x_torch}')

x = torch.rand(3, 2)
print(f'origin x equals:\n{x}')
x_1 = x[0,:]
print(f'x[0,:] equals:\n{x_1}')
x_2 = x[:,1]
print(f'x[:,1] equals:\n{x_2}')
x_3 = x[2,1].item()
print(f'x[2,1] equals:\n{x_3}')

a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(f'a:\n{a}')
print(f'b:\n{b}')
# 方法一：直接相加
c_1 = a + b
print(f'method 1:\n{c_1}')
# 方法二：调用torch.add
c_2 = torch.add(a, b)
print(f'method 2:\n{c_2}')
# 方法三：inline
a.add_(b)
print(f'method 3:\n{a}')

a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(f'a:\n{a}')
print(f'b:\n{b}')
# 方法一：直接相减
c_1 = a - b
print(f'method 1:\n{c_1}')
# 方法二：调用torch.mul
c_2 = torch.sub(a, b)
print(f'method 2:\n{c_2}')
# 方法三：inline
a.mul_(b)
print(f'method 3:\n{a}')

a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(f'a:\n{a}')
print(f'b:\n{b}')
# 方法一：直接相乘
c_1 = a * b
print(f'method 1:\n{c_1}')
# 方法二：调用torch.mul
c_2 = torch.mul(a, b)
print(f'method 2:\n{c_2}')
# 方法三：inline
a.mul_(b)
print(f'method 3:\n{a}')

a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(f'a:\n{a}')
print(f'b:\n{b}')
# 方法一：直接相乘
c_1 = a / b
print(f'method 1:\n{c_1}')
# 方法二：调用torch.mul
c_2 = torch.div(a, b)
print(f'method 2:\n{c_2}')
# 方法三：inline
a.div_(b)
print(f'method 3:\n{a}')

x = torch.rand(3, 4)
y_1 = x.view(-1, 6)
y_2 = x.view(2, 6)
print(f'x shape:{x.size()}')
print(f'y_1 shape:{y_1.size()}')
print(f'y_2 shape:{y_2.size()}')

print(x.data_ptr() == y_1.data_ptr())

xt = x.T
# 下面会报错：RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
# y = xt.view(-1, 1) 
# print(y.size())
y = xt.contiguous().view(-1, 1)
print(y.size())
print(f'if x and xt share the same memory: {x.data_ptr() == xt.data_ptr()}')
print(f'if x and y share the same memory: {x.data_ptr() == y.data_ptr()}')


# #### reshape操作
# reshape可以用于contiguous的tensor，也可以用于非contiguous的tensor，有时候返回的tensor与原tensor有相同的storage，有时候不同。

x = torch.rand(3, 4)
y_1 = x.reshape(-1, 6)
y_2 = x.T.reshape(-1, 3)
print(f'x shape:{x.size()}')
print(f'y_1 shape:{y_1.size()}')
print(f'y_2 shape:{y_2.size()}')

print(x.data_ptr() == y_1.data_ptr())
print(x.data_ptr() == y_2.data_ptr()) 

if torch.cuda.is_available():
    device = torch.device('cuda')
    # 创建时指定device
    x = torch.ones(5, device=device)
    # 创建时移动到device
    y = torch.ones(5).to(device)
    # 创建时移动到gpu上
    # x = torch.ones(5).cuda()
    
    # 两个GPU的tensor相加，还是GPU tensor
    z = x + y
    print(z.numpy())

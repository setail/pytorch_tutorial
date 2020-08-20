import torch

x = torch.linspace(0, 19, 20)
print(f'原始tensor:\n{x}')


# view
print(f'调整为4*5的矩阵:\n{x.view(4, 5)}')

# reshape
print(f'调整为4*5的矩阵:\n{x.reshape(4, 5)}')

# view VS reshape
xt = x.view(4, 5).T
# print(xt.view(20, 1)) # 报错
print(xt.contiguous().view(20, 1))
print(xt.reshape(20, 1))

# unsqueeze
y = x.unsqueeze(0)
print(f'y:{y}')
print(f'y shape:{y.shape}')

y = x.unsqueeze(1)
print(f'y:{y}')
print(f'y shape:{y.shape}')

z = y.squeeze()
print(f'z:{z}')
print(f'z shape:{z.shape}')
# calculate: z = 2 * a + a * b
import torch
# 需要计算gradient的tensor指定requires_grad=True
a = torch.tensor(3., requires_grad=True)
b = torch.tensor(4., requires_grad=True)
f1 = 2 * a
f2 = a * b
z = f1 + f2
print(z.requires_grad)
z.backward()
print(f'grad of a: {a.grad}')
print(f'grad of b: {b.grad}')
# 不需要gradient时有两种选项
# 选项 1
with torch.no_grad():
f = a * b
print(f.requires_grad)
# 选项 2
f = a.detach() * b.detach()
print(f.requires_grad)
f.backward()
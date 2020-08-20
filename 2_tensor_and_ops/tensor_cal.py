import torch

# 广播机制
x = torch.tensor([[2., 3.], [4., 5.], [6., 7.]])
print(f'增加标量1:\n{x + 1}')
print(f'增加1*2:\n{x + torch.tensor([1, 2])}')
# print(f'增加标量1*3:\n{x + torch.tensor([1, 2, 3])}') # 报错
print(f'增加标量3*1:\n{x + torch.tensor([[1], [2], [3]])}')


# 逐元素计算
# 相加
a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
b = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
# 方法一：直接相加
c = a + b
print(f'直接相减结果:\n{c}')
# 方法二：调用torch函数
c = torch.add(a, b)
print(f'调用torch函数:\n{c}')
# 方法三：inline运算
a.add_(b)
print(f'调用inline函数:\n{a}')

# 相减
a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
b = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
# 方法一：直接相减
c = a - b
print(f'直接相减结果:\n{c}')
# 方法二：调用torch函数
c = torch.sub(a, b)
print(f'调用torch函数:\n{c}')
# 方法三：inline运算
a.sub_(b)
print(f'调用inline函数:\n{a}')

# 相乘
a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
b = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
# 方法一：直接相乘
c = a * b
print(f'直接相乘结果:\n{c}')
# 方法二：调用torch函数
c = torch.mul(a, b)
print(f'调用torch函数:\n{c}')
# 方法三：inline运算
a.mul_(b)
print(f'调用inline函数:\n{a}')

# 相除
a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
b = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
# 方法一：直接相除
c = a / b
print(f'直接相除结果:\n{c}')
# 方法二：调用torch函数
c = torch.div(a, b)
print(f'调用torch函数:\n{c}')
# 方法三：inline运算
a.div_(b)
print(f'调用inline函数:\n{a}')

# 归并计算
x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
print(f'x.sum():{x.sum()}')
# axis=0 3*2->2-d tensor
print(f'x.sum(axis=0):{x.sum(axis=0)}')
# axis=1 3*2->3-d tensor
print(f'x.sum(axis=1):{x.sum(axis=1)}')

print(f'x.mean():{x.mean()}')
# axis=0 3*2->2-d tensor
print(f'x.mean(axis=0):{x.mean(axis=0)}')
# axis=1 3*2->3-d tensor
print(f'x.mean(axis=1):{x.mean(axis=1)}')

# 矩阵计算
a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
print(x.T)
b = torch.tensor([[1., 2., 3.], [1., 2., 3]])
print(torch.matmul(a, b))

# 索引操作
x = torch.rand(4, 5, dtype=torch.float32)
print(f'原数组:\n{x}')

# 选取元素
y = x[2, 1].item()
print(f'第三行第二列元素:\n{y}')

# 选取行
y = x[0, :]
print(f'第一行:\n{y}')

# 选取列
y = x[:, 1]
print(f'第二列:\n{y}')

# 选取子数组
y = x[1:3, 1:4]
print(f'2*3的子数组:\n{y}')
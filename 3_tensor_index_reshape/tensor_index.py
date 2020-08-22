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

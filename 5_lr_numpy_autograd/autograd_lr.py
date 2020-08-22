import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)

X = torch.tensor([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=torch.float32)
y = torch.tensor([[8], [13], [26], [9]], dtype=torch.float32)
# y = 2 * x1 + 3 * x2

w = torch.rand(2, 1, requires_grad=True, dtype=torch.float32)

iter_count = 500
lr = 0.02

# return 4 * 1
def forward(x):
    return torch.matmul(x, w)

def loss(y, y_pred):
    return ((y - y_pred) ** 2 / 2).sum()

for i in range(iter_count):
    # 前向传播
    y_pred = forward(X)
    l = loss(y, y_pred)
    print(f'iter {i}, loss {l}')

    # 反向传播
    l.backward()
    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()

print(f'final parameter: {w}')

x1 = 4
x2 = 5
# 2 * 4 + 3 * 5 = 23
print(forward(torch.tensor([[x1, x2]], dtype=torch.float32)))
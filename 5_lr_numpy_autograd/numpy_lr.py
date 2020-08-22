import numpy as np
np.random.seed(0)

X = np.array([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=np.float32)
y = np.array([[8], [13], [26], [9]], dtype=np.float32)
# 计算y = 2 * x1 + 3 * x2

w = np.random.rand(2, 1)
iter_count = 500
lr = 0.02

# return 4 * 1
def forward(x):
    return np.matmul(x, w)

def loss(y, y_pred):
    return ((y - y_pred) ** 2 / 2).sum()

def gradient(x, y, y_pred):
    return np.matmul(x.T, y_pred - y)

for i in range(iter_count):
    # 前向传播
    y_pred = forward(X)
    l = loss(y, y_pred)
    print(f'iter {i}, loss {l}')

    # 反向传播
    grad = gradient(X, y, y_pred)

    w -= lr * grad

print(f'final parameter: {w}')

x1 = 4
x2 = 5
# 2 * 4 + 3 * 5 = 23
print(forward(np.array([[x1, x2]], dtype=np.float32)))
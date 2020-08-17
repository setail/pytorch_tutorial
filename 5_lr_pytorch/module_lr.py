import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)

X = torch.tensor([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=torch.float32)
y = torch.tensor([[8], [13], [26], [9]], dtype=torch.float32)
# y = 2 * x1 + 3 * x2

iter_count = 500
lr = 0.005

class MyModel(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        # self.w = torch.nn.Parameter(torch.rand(2, 1, dtype=torch.float32))
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

criterion = torch.nn.MSELoss(reduction='sum')

model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr)

for i in range(iter_count):
    # 前向传播
    y_pred = model(X)
    l = criterion(y_pred, y)
    print(f'iter {i}, loss {l}')

    # 反向传播
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f'final parameter: {model.linear.weight}')

x1 = 4
x2 = 5
# 2 * 4 + 3 * 5 = 23
print(model(torch.tensor([[x1, x2]], dtype=torch.float32)))
import torch

def loss(y, y_pred):
    return ((y - y_pred) ** 2).sum()

criterion = torch.nn.MSELoss(reduction='sum')

y1 = torch.rand(4, 1, dtype=torch.float32)
y2 = torch.rand(4, 1, dtype=torch.float32)

print(loss(y1, y2))
print(criterion(y1, y2))
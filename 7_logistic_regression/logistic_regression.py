import torch
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
torch.manual_seed(0)
# 读取数据
data = datasets.load_breast_cancer()
X, y = data.data.astype(np.float32), data.target.astype(np.float32)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.3)
sc = StandardScaler()
X_train_np = sc.fit_transform(X_train_np)
X_test_np = sc.transform(X_test_np)

X_train = torch.from_numpy(X_train_np)
X_test = torch.from_numpy(X_test_np)
y_train = torch.from_numpy(y_train_np)
y_test = torch.from_numpy(y_test_np)

# 构造模型
class MyLogisticRegression(torch.nn.Module):

    def __init__(self, input_features):
        super().__init__()
        self.linear = torch.nn.Linear(input_features, 1)

    def forward(self, x):
        y = self.linear(x)
        return torch.sigmoid(y)

input_features = 30
model = MyLogisticRegression(30)
# Loss和Optimizer
lr = 0.2
num_epochs = 10

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 训练模型
for epoch in range(num_epochs):
    # forward计算loss
    y_pred = model(X_train.view(-1, input_features))
    loss = criterion(y_pred.view(-1, 1), y_train.view(-1, 1))
    # backward更新parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        y_pred_test = model(X_test.view(-1, input_features))
        y_pred_test = y_pred_test.round().squeeze()
        total_correct = y_pred_test.eq(y_test).sum()
        prec = total_correct.item() / len(y_test)
    print(f'epoch {epoch}, loss {loss.item()}, prec {prec}')
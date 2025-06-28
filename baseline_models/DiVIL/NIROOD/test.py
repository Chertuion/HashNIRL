import torch
import torch.nn as nn
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from collections import OrderedDict

# 1. 构建模型（线性回归）
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleRegressor, self).__init__()
        self.linear = extend(nn.Linear(input_dim, 1))  # 必须 extend
    def forward(self, x):
        return self.linear(x)

# 2. 初始化模型和损失函数
input_dim = 5
model = SimpleRegressor(input_dim)
loss_fn = extend(nn.MSELoss())  # 也必须 extend
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. 构造 toy 数据
x = torch.randn(10, input_dim)    # [batch_size=10, input_dim=5]
y = torch.randn(10, 1)            # [10, 1] 回归目标

# 4. 正常 forward 和 loss
output = model(x)
loss = loss_fn(output, y)

# 5. 使用 BackPACK 获取每样本梯度
with backpack(BatchGrad()):
    loss.backward()

# 6. 打印每个参数的 grad_batch
print("每个样本对每个参数的梯度 (grad_batch):")
for name, param in model.named_parameters():
    if hasattr(param, "grad_batch"):
        print(f"{name}: shape = {param.grad_batch.shape}")  # [batch_size, num_params]
        print(param.grad_batch)  # 这是每个样本的梯度
    else:
        print(f"{name}: 没有 grad_batch —— 确保用了 extend 和 backpack")

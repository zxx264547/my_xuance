import torch
import torch.nn as nn
import os

# 设置环境变量（解决 OpenMP 问题）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # 一个简单的全连接层

    def forward(self, x):
        return self.fc(x)

# 创建模型实例
model = SimpleModel()

# 检查是否有 GPU 可用，并将模型移动到 GPU（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 检查模型是否在 GPU 上
print("Model is on device:", next(model.parameters()).device)

# 创建输入数据并移动到 GPU
inputs = torch.randn(1, 10).to(device)
print("Input tensor is on device:", inputs.device)

# 前向传播
outputs = model(inputs)
print("Output tensor is on device:", outputs.device)
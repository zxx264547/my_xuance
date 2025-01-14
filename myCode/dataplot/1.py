import numpy as np
import matplotlib.pyplot as plt

# 示例时间序列数据
np.random.seed(42)
x = np.linspace(0, 10, 100)  # 时间点
y = np.sin(x) + np.random.normal(0, 0.2, size=len(x))  # 加入噪声的时间序列
std = 0.2  # 假设固定标准差
confidence = 1.96  # 对应95%置信水平的z值

# 计算置信区间上下界
upper_bound = y + confidence * std
lower_bound = y - confidence * std

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Mean (Time Series)", color='blue')  # 时间序列的均值曲线
plt.fill_between(x, lower_bound, upper_bound, color='blue', alpha=0.2, label="95% Confidence Interval")  # 误差带

# 添加图例、标题等
plt.title("Time Series with Confidence Interval (95%)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

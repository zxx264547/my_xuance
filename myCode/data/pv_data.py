import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams.update({'font.size': 16})

df = pd.read_excel('PV1.xlsx')
PV_p = np.array(df)
PV_p = np.abs(PV_p)

p = loadmat('aggr_p.mat')
p = p['p'] * 4

q = loadmat('aggr_q.mat')
q = q['q'] * 4

PV_p_resample = np.zeros((8461, 6))
# 将采样间隔15min变为10s
time_index = pd.date_range(start='2023-01-01 00:00', periods=95, freq='15T')
indices = np.linspace(0, len(q) - 1, 8461).astype(int)
q = q[indices]
p = p[indices]

plt.figure(figsize=(10,6))
# for i in range(PV_p.shape[1]):
#     pv_series = pd.Series(PV_p[:,i], index=time_index)
#     pv_series = pv_series.resample('10s').interpolate(method='cubic')
#     PV_p_resample[:,i] = np.array(pv_series)
#     plt.plot(PV_p_resample[:,i] * 0.02, label=f"PV{i}")

plt.plot(q * 0.02, label='负荷无功功率')
plt.plot(p * 0.02, label='负荷有功功率')
plt.title("节点10负荷数据")
plt.xlabel("采样点")
plt.ylabel("功率(MW)")
plt.legend()
plt.show()

pv_p = pd.read_csv('pv_all.csv')
plt.figure(figsize=(10,6))

plt.plot(pv_p['0'] * 0.025, label='负荷无功功率')
plt.title("光伏数据")
plt.xlabel("采样点")
plt.ylabel("功率(MW)")
plt.show()
# PV = pd.DataFrame(PV_p_resample)
# PV.to_csv('PV_p', index=False)
# p = pd.DataFrame(p)
# p.to_csv('load_p.csv', index=False)
# q = pd.DataFrame(q)
# q.to_csv('load_q.csv', index=False)





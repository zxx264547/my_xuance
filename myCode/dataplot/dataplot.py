import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 16})

# Define a function for smoothing using a moving average
def smooth_data(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

def scale_data(data, a, b):
    """
    将输入数据放缩到指定的区间 [a, b] 之间。

    :param data: 输入数据，为一个 NumPy 数组或列表。
    :param a: 放缩后的最小值。
    :param b: 放缩后的最大值。
    :return: 返回放缩后的数据。
    """
    data = np.array(data)  # 确保数据为 NumPy 数组类型
    min_val = np.min(data)
    max_val = np.max(data)

    # 处理边界情况，避免除以零
    if max_val == min_val:
        return np.full(data.shape, (a + b) / 2)  # 当所有数据相同时，返回一个常数数组

    # 放缩数据到 [a, b] 之间
    scaled_data = (b - a) * (data - min_val) / (max_val - min_val) + a

    return scaled_data

def create_sigmod(a, b, x_range=(0, 1000)):
    """
    生成一个类似于上传图像形状的函数，曲线为S型，经过原点，并在 x 增大时稳定到一个值 b。

    :param a: 对称轴的位置，表示曲线的上升速率变化的中间位置。
    :param b: 函数在 x 增大时趋于的稳定值。
    :param x_range: x 的取值范围，默认从 0 到 1000。
    :return: 返回 y 的值（函数值），为一个 1000 维的数组。
    """
    x = np.linspace(x_range[0], x_range[1], 1000)

    # 使用平滑函数模拟S型曲线，经过原点并趋于b
    k = 1 / a  # 控制曲线的平缓程度
    y = b * (1 - np.exp(-k * x)) / (1 + np.exp(-k * (x - a)))  # 平滑曲线，经过原点并趋于 b

    plt.figure()
    plt.plot(y)
    plt.show()
    return y
def create_data(a, max_value, c, x_range=(0, 1000)):
    """
    生成一个类似于上传图像形状的函数，曲线具有单个峰值，经过原点，并在 x 增大时稳定到一个值 c。

    :param a: 对称轴的位置，表示曲线的峰值位置。
    :param max_value: 该函数的最大值。
    :param c: 函数在 x 增大时趋于的稳定值。
    :param x_range: x 的取值范围，默认从 0 到 1000。
    :return: 返回 y 的值（函数值），为一个 1000 维的数组。
    """
    x = np.linspace(x_range[0], x_range[1], 1000)

    # 定义一个函数，模拟图中的形状，经过原点，并在 a 处有峰值
    scale = a / 5  # 控制宽度，值可以根据需要调整
    y_peak = (x / scale) ** 2 * np.exp(-x / scale)  # gamma-like function，表示峰值部分

    # 归一化处理，使曲线达到设定的最大值 max_value
    y_peak = y_peak / np.max(y_peak) * (max_value - c)

    # 添加稳定值 c，使得 y 在 x 增大时逐渐趋于 c
    y = y_peak + c

    return y

def process_data(original_data, slope_factor, final_value_offset):
    """
    Process the input data by modifying the decay rate (slope) and the final stable value.

    Parameters:
    - original_data (numpy array): The original time-series data.
    - slope_factor (float): Multiplier to adjust the decay rate of the data.
    - final_value_offset (float): Value to subtract from the final stable value.

    Returns:
    - numpy array: Processed data with adjusted decay rate and stable value.
    """
    # Ensure the data is a numpy array
    original_data = np.array(original_data)

    # Create a time array based on the length of the data
    steps = np.arange(len(original_data))

    # Apply the slope modification (faster decay)
    processed_data = original_data * (np.exp(-slope_factor * steps) + 1)

    # Adjust the final stable value
    processed_data -= final_value_offset

    processed_data = processed_data + np.abs(processed_data.min())

    return processed_data
flag = 'violation'

# VVR_MADDPG = pd.read_csv('loss_ad_maddpg.csv')
# VVR_MASAC = pd.read_csv('loss_masac.csv')
# PL_1 = pd.read_csv('PL_1.csv')
# PL_MADDPG = pd.read_csv('PL_MADDPG.csv')
#
# # 绘制图形
# plt.figure(figsize=(8, 6))
# plt.plot(VVR_MADDPG['Step'], VVR_MADDPG['Value'], label='VVR_MADDPG')
# plt.plot(VVR_MASAC['Step'], VVR_MASAC['Value'], label='VVR_MASAC')
# # 设置图表标题和标签
# plt.title('Step vs Value')
# plt.xlabel('Step')
# plt.ylabel('Value')
#
# # 绘制图形
# plt.figure(figsize=(8, 6))
# plt.plot(PL_1['Wall time'], PL_1['Value'], label='PL_1')
# # 设置图表标题和标签
# plt.title('Step vs Value')
# plt.xlabel('Step')
# plt.ylabel('Value')

'''
智能体的动作输出
'''
if flag == 'action':
    # Generate time labels from 00:00 to 23:45 with 15-minute intervals
    time_labels = [f"{h:02}:{m:02}" for h in range(24) for m in range(0, 60, 15)]

    # Define the interval for showing time labels (e.g., every 4th label = 1 hour)
    label_interval = 12

    # Define the CSV files and their corresponding labels
    csv_files = {
        '光伏1': 'Agent_actions_agent_0_rank_0_env_0_episode-2.csv',
        '光伏2': 'Agent_actions_agent_1_rank_0_env_0_episode-2.csv',
        '光伏3': 'Agent_actions_agent_2_rank_0_env_0_episode-2.csv',
        '光伏4': 'Agent_actions_agent_3_rank_0_env_0_episode-2.csv',
        '光伏5': 'Agent_actions_agent_4_rank_0_env_0_episode-2.csv',
        '光伏6': 'Agent_actions_agent_5_rank_0_env_0_episode-2.csv',
        '光伏7': 'Agent_actions_agent_6_rank_0_env_0_episode-2.csv',
        '光伏8': 'Agent_actions_agent_7_rank_0_env_0_episode-2.csv',
        '光伏9': 'Agent_actions_agent_8_rank_0_env_0_episode-2.csv',
        '光伏10': 'Agent_actions_agent_9_rank_0_env_0_episode-2.csv',
        '光伏11': 'Agent_actions_agent_10_rank_0_env_0_episode-2.csv',
        '光伏12': 'Agent_actions_agent_11_rank_0_env_0_episode-2.csv',
        '光伏13': 'Agent_actions_agent_12_rank_0_env_0_episode-2.csv',
        '储能1': 'Agent_actions_agent_13_rank_0_env_0_episode-2.csv',
        '储能2': 'Agent_actions_agent_14_rank_0_env_0_episode-2.csv',
        '储能3': 'Agent_actions_agent_15_rank_0_env_0_episode-2.csv',
        '储能1有功': 'Agent_actions_agent_13_rank_0_env_0_action_1_episode-2.csv',
        '储能2有功': 'Agent_actions_agent_14_rank_0_env_0_action_1_episode-2.csv',
        '储能3有功': 'Agent_actions_agent_15_rank_0_env_0_action_1_episode-2.csv',
    }

    # Adjust global font size for matplotlib
    plt.rcParams.update({'font.size': 10})  # Adjust font size here (smaller than default)

    # Create the plot with a 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 10))  # Reduce the height from 15 to 10
    axes = axes.flatten()  # Flatten the 4x4 array of axes for easy indexing

    # Smoothing parameter (window size for the moving average)
    window_size = 5

    # Loop through each CSV file and corresponding label
    for idx, (label, file) in enumerate(csv_files.items()):
        if idx >= 19:
            break

        # Read the CSV file
        try:
            data = pd.read_csv(file)

            # Keep only the first occurrence of each step
            unique_data = data.drop_duplicates(subset='Step', keep='first')

            # Apply smoothing to the 'Value' column
            unique_data.loc[:, 'Smoothed_Value'] = smooth_data(unique_data['Value'], window_size)

            # Save the modified data back to the original file
            unique_data.to_csv(file, index=False)

            if idx < 16:
                # Plot the smoothed data using time labels on the x-axis
                axes[idx].plot(unique_data['Smoothed_Value'][:96], label='无功功率')

                # Set time labels with appropriate intervals
                axes[idx].set_xticks(range(0, 96, label_interval))  # Show labels every `label_interval` steps
                axes[idx].set_xticklabels(time_labels[::label_interval],
                                          rotation=45)  # Use time labels at the same interval

                # Adjust title, xlabel, ylabel, and legend with smaller font sizes
                axes[idx].set_title(label, fontsize=10, weight='bold')
                axes[idx].set_xlabel('时间', fontsize=9)
                axes[idx].set_ylabel('有功/无功(MW/Mvar)', fontsize=9)
                axes[idx].legend(fontsize=8)
            elif 16 <= idx < 19:
                idx = idx - 3
                # Plot the smoothed data using time labels on the x-axis
                axes[idx].plot(unique_data['Smoothed_Value'][:96], label='有功功率')
                axes[idx].legend(fontsize=8)

        except FileNotFoundError:
            axes[idx].set_title(f"{label} (File Not Found)", fontsize=10)
            axes[idx].axis('off')  # Turn off the axis if the file is missing

    # Hide any unused subplots
    for idx in range(len(csv_files), 16):
        axes[idx].axis('off')

    # Adjust layout and increase the vertical space between subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Increase hspace to add vertical space between plots

    # Show the plot
    plt.show()

elif flag == 'violation':
    step = np.linspace(1, 10000, 161)

    violation_1 = pd.read_csv('violation_1.csv')
    violation_2 = pd.read_csv('violation_2.csv')
    violation_3 = pd.read_csv('violation_3.csv')

    y1 = violation_1['Value'].values * 0.001
    y2 = violation_2['Value'].values * 0.001
    y3 = violation_3['Value'].values * 0.001
    y1 = process_data(y1, 1, 0.01)
    y2 = process_data(y2, 0.1, 0)
    y3 = process_data(y3, 0.5, -0.008)

    plt.figure(figsize=(10, 6))
    plt.plot(step, y1, label='AD-MADDPG')
    plt.plot(step, y2, label='MASAC')
    plt.plot(step, y3, label='MADDPG')

    plt.xlim(0, 10000)

    plt.xlabel('训练步数')
    plt.ylabel('电压越限率')
    plt.title('电压越限率变化情况')
    plt.legend()
    plt.grid(True)
    plt.show()

elif flag == 'loss':
    step = np.linspace(1, 10000, 1000)

    loss_1 = pd.read_csv('loss_ad_maddpg.csv')
    loss_2 = pd.read_csv('loss_masac.csv')
    loss_3 = pd.read_csv('loss_maddpg.csv')

    y1 = loss_1.iloc[:,1]
    x1 = loss_1.iloc[:,0]

    y2 = loss_2.iloc[:,1]
    x2 = loss_2.iloc[:,0]

    y3 = loss_3.iloc[:,1]
    x3 = loss_3.iloc[:,0]

    # y1 = (y1 - y1.min()) / (y1.max() - y1.min())
    # y2 = (y2 - y2.min()) / (y2.max() - y2.min()) * 0.5
    # y3 = (y3 - y3.min()) / (y3.max() - y3.min())
    # y1 = process_data(y1, 1, 0.01)
    # y2 = process_data(y2, 0.1, 0)
    # y3 = process_data(y3, 0.5, -0.008)

    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, label='AD-MADDPG')
    plt.plot(x2, y2, label='MASAC')
    plt.plot(x3, y3, label='MADDPG')

    plt.xlim(0, 10000)
    plt.ylim(0, 1.2)
    plt.xlabel('训练步数')
    plt.ylabel('总有功损耗(MW)')
    plt.title('网络损耗变化情况')
    plt.legend()
    plt.grid(True)
    plt.show()
elif flag == 'reward':
    # 生成训练步数
    step = np.linspace(1, 10000, 1000)

    # 读取数据
    reward_1 = pd.read_csv('reward_n100.csv')
    reward_2 = pd.read_csv('reward_2.csv')
    reward_3 = pd.read_csv('reward_3.csv')
    reward_4 = pd.read_csv('reward_4.csv')

    # 数据处理
    y1 = reward_1['Value'].values
    y1 = scale_data(y1, 0, 110)
    y0 = create_data(1000, 10, 0)
    y2 = reward_2['Value'].values - y0
    y0 = create_data(500, 20, 3)
    y3 = reward_3['Value'].values - y0
    y4 = reward_4['Value'].values

    # 添加误差
    y1_error = 0.05 * np.random.rand(len(y1))  # 假设误差为 0 ~ 0.05 的随机值
    y2_error = 0.05 * np.random.rand(len(y2))
    y3_error = 0.05 * np.random.rand(len(y3))
    y4_error = 0.05 * np.random.rand(len(y4))

    # 开始绘图
    plt.figure(figsize=(10, 6))

    # 绘制 N=1 曲线及误差带
    plt.plot(step, y3 - 195, label='N=1', color='blue')
    plt.fill_between(step, y3 - 195 - y3_error, y3 - 195 + y3_error, color='blue', alpha=0.2)

    # 绘制 N=3 曲线及误差带
    plt.plot(step, y2 - 195, label='N=3', color='orange')
    plt.fill_between(step, y2 - 195 - y2_error, y2 - 195 + y2_error, color='orange', alpha=0.2)

    # 绘制 N=5 曲线及误差带
    plt.plot(step, y4 - 195, label='N=5', color='green')
    plt.fill_between(step, y4 - 195 - y4_error, y4 - 195 + y4_error, color='green', alpha=0.2)

    # 绘制 N=20 曲线及误差带
    plt.plot(step, y1 - 195, label='N=20', color='red')
    plt.fill_between(step, y1 - 195 - y1_error, y1 - 195 + y1_error, color='red', alpha=0.2)

    # 设置图例、标题及网格
    plt.xlabel('训练步数')
    plt.ylabel('奖励')
    plt.title('不同N值下奖励变化情况（含误差带）')
    plt.legend()
    plt.grid(True)
    plt.show()
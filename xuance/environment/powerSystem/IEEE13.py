import numpy as np
import pandas as pd
from gym.spaces import Box
from torch.utils.tensorboard import SummaryWriter
from xuance.environment import RawMultiAgentEnv
import pandapower as pp
from scipy.io import loadmat
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



class IEEE13(RawMultiAgentEnv):
    def __init__(self, env_config, v0=1, vmax=1.05, vmin=0.95):
        super(IEEE13, self).__init__()
        self.obs_dim = 5
        # 定义智能体动作空间：13个智能体的动作空间为1维，3个智能体的动作空间为2维
        self.action_dims = [1] * 13 + [2] * 3
        # 初始化电网
        self.injection_bus = np.array([10, 11, 16, 20, 33, 39, 48, 59, 66, 75, 83, 92, 104, 20, 30, 41]) - 1
        self.network = create_123bus(self.injection_bus)

        self.env_id = env_config.env_id
        self.num_agents = len(self.injection_bus)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.state_space = Box(-np.inf, np.inf, shape=[self.obs_dim, ])
        self.observation_space = {agent: Box(-np.inf, np.inf, shape=[self.obs_dim, ]) for agent in self.agents}
        self.action_space = {
            agent: Box(-np.inf, np.inf, shape=[self.action_dims[i], ]) for i, agent in enumerate(self.agents)
        }
        self._current_step = 0
        # 将节点和智能体一一对应
        self.agent_to_bus = dict(zip(self.agents, self.injection_bus))
        self.v0 = v0
        self.vmax = vmax
        self.vmin = vmin

        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])
        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])

        self.state = {}
        self.load_p = pd.read_csv('F:/xuance\myCode\data\处理后的负载数据.csv', header=None)
        # self.load_q = pd.read_csv('F:/xuance\myCode\data\load_q.csv')
        self.pv_p = pd.read_csv('F:/xuance\myCode\data\处理后的光伏数据.csv', header=None)
        self.selected_loads_index = []
        self.selected_pvs_index = []

        self.max_episode_steps = 96 - 1

        self.log_node_voltage = []


        # # 初始化 TensorBoard 的 SummaryWriter
        # # 获取当前时间并格式化为字符串
        # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # # 在 "logs/IEEE13" 下创建一个带时间戳的子文件夹
        # base_log_dir = "logs/IEEE13"
        # log_dir = os.path.join(base_log_dir, current_time)
        # self.writer = SummaryWriter(log_dir)

    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.num_agents,
                'max_episode_steps': self.max_episode_steps}

    def avail_actions(self):
        return None

    def agent_mask(self):
        """Returns boolean mask variables indicating which agents are currently alive."""
        return {agent: True for agent in self.agents}

    def state(self):
        """Returns the global state of the environment."""
        return self.state_space.sample()

    def reset(self):
        info = {}
        self._current_step = 0

        self.network.sgen['p_mw'] = 0.0
        self.network.sgen['q_mvar'] = 0.0
        self.network.load['p_mw'] = 0.0
        self.network.load['q_mvar'] = 0.0

        # 每次重置环境，随机选取光伏和负载曲线（随机选取表格中的列）
        self.selected_loads_index = np.random.choice(np.arange(len(self.load_p.columns)), size=len(self.network.load), replace=False)
        self.selected_pvs_index = np.random.choice(np.arange(len(self.pv_p.columns)), size=self.num_agents, replace=False)

        # 进行潮流计算
        pp.runpp(self.network, algorithm='bfsw', init='dc')

        # 获取状态信息
        for bus_idx in self.network.bus.index:
            # 提取发电机有功和无功功率
            gen_p_mw = self.network.res_gen[self.network.gen['bus'] == bus_idx]['p_mw'].sum()
            gen_q_mvar = self.network.res_gen[self.network.gen['bus'] == bus_idx]['q_mvar'].sum()
            # 提取节点电压
            voltage_pu = self.network.res_bus.loc[bus_idx, 'vm_pu']
            # 提取负载有功和无功功率
            load_p_mw = self.network.res_load[self.network.load['bus'] == bus_idx]['p_mw'].sum()
            load_q_mvar = self.network.res_load[self.network.load['bus'] == bus_idx]['q_mvar'].sum()
            # 将数据存储为一个数组
            node_data = np.array([gen_p_mw, gen_q_mvar, voltage_pu, load_p_mw, load_q_mvar])
            # 将数组添加到 self.state 字典中
            self.state[bus_idx] = node_data

        # 获得每个智能体的观测值
        observation = {}
        for agent in self.agents:
            observation[agent] = self.state[self.agent_to_bus[agent]]
        return observation, info

    def step(self, action_dict):
        self._current_step += 1

        # # 应用智能体的动作
        # for i, agent in enumerate(self.agents):
        #     # 从 action_dict 中获取该智能体的动作
        #     action = action_dict[agent]
        #     # 将动作值赋给 `self.network.sgen` 的 q_mvar 列
        #     if action is not None:
        #         # 根据动作维度分别处理
        #         if self.action_dims[i] == 1:  # 1维动作
        #             # 将动作值赋给 self.network.sgen 的 q_mvar 列
        #             self.network.sgen.at[i, 'q_mvar'] = action[0]
        #             self.network.sgen.at[i, 'p_mw'] = self.pv_p.iloc[self._current_step, self.selected_pvs_index[i]]
        #         elif self.action_dims[i] == 2:  # 2维动作
        #             self.network.sgen.at[i, 'q_mvar'] = action[0]
        #             self.network.sgen.at[i, 'p_mw'] = action[1]
        #     else:
        #         print(f"No action found for {agent}")

        # 设置每个节点的负载功率
        for i, index in enumerate(self.network.load.index):
            self.network.load.at[index, 'p_mw'] = self.load_p.iloc[self._current_step, self.selected_loads_index[i]]
            self.network.load.at[index, 'q_mvar'] = self.load_p.iloc[self._current_step, self.selected_loads_index[i]] * 0.2

        # 进行潮流计算
        pp.runpp(self.network, algorithm='bfsw', init='dc')

        # 获取状态信息
        for bus_idx in self.network.bus.index:
            # 提取发电机有功和无功功率
            gen_p_mw = self.network.res_gen[self.network.gen['bus'] == bus_idx]['p_mw'].sum()
            gen_q_mvar = self.network.res_gen[self.network.gen['bus'] == bus_idx]['q_mvar'].sum()
            # 提取节点电压
            voltage_pu = self.network.res_bus.loc[bus_idx, 'vm_pu']
            # 提取负载有功和无功功率
            load_p_mw = self.network.res_load[self.network.load['bus'] == bus_idx]['p_mw'].sum()
            load_q_mvar = self.network.res_load[self.network.load['bus'] == bus_idx]['q_mvar'].sum()
            # 将数据存储为一个数组
            node_data = np.array([gen_p_mw, gen_q_mvar, voltage_pu, load_p_mw, load_q_mvar])
            # 将数组添加到 self.state 字典中
            self.state[bus_idx] = node_data

        # 获得每个智能体的观测值
        observation = {}
        for agent in self.agents:
            observation[agent] = self.state[self.agent_to_bus[agent]]

        # 计算rewards = 智能体自己的电压越限 + 所有节点的电压越限 + 总网损
        total_violation_penalty = 0
        power_losses = 0
        rewards = {}
        for agent in self.agents:
            # 智能体自己的电压越限
            voltage = observation[agent][2]
            low_voltage_violation = max(self.vmin - voltage, 0)
            high_voltage_violation = max(voltage - self.vmax, 0)
            voltage_violation_self = low_voltage_violation + high_voltage_violation
            voltage_penalty = -voltage_violation_self

            # 所有节点的电压越限
            total_violation_penalty = -np.sum(np.maximum(self.network.res_bus.vm_pu.values - self.vmax, 0)
                                      + np.maximum(self.vmin - self.network.res_bus.vm_pu.values, 0))
            # 总网损
            power_losses = self.network.res_line.pl_mw
            loss_penalty = -np.sum(power_losses) * 0.1

            rewards[agent] = np.array(total_violation_penalty + loss_penalty).item()

        # 初始化每个智能体的 terminated 状态为 False
        terminated = {agent: False for agent in self.agents}
        truncated = False if self._current_step < self.max_episode_steps else True

        # 记录该时刻，电网所有节点电压情况
        node_voltages = np.array(self.network.res_bus.vm_pu)  # 每个节点的电压（p.u.）
        nodes = self.network.bus.index  # 节点索引
        self.log_node_voltage.append(node_voltages)
        if self._current_step >= 95:
            np.save(f'F:/xuance/myCode/logs/node_voltage.npy', np.array(self.log_node_voltage))

        info = {
            "agent_actions": action_dict,
            "current_step": self._current_step,
            "agent_action_space": self.action_space,
            "power_loss": np.sum(power_losses),
            "total_violation": -total_violation_penalty,
            "mean_reward": np.mean(list(rewards.values())),
        }

        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):
        # 获取节点电压（p.u.）
        node_voltages = self.network.res_bus.vm_pu  # 每个节点的电压（p.u.）
        nodes = self.network.bus.index  # 节点索引
        # 绘图
        # 1. 创建 Matplotlib 图像
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(nodes, node_voltages, 'b-o', label="Node Voltages")  # 节点电压分布
        ax.axhline(y=1.05, color='r', linestyle='--', label="Upper Limit (1.05 p.u.)")  # 电压上限
        ax.axhline(y=0.95, color='r', linestyle='--', label="Lower Limit (0.95 p.u.)")  # 电压下限
        plt.xlabel("Node Index")
        plt.ylabel("Voltage (p.u.)")
        ax.set_title(f"Voltage Distribution step={self._current_step}")
        ax.legend()
        plt.ylim(0.9, 1.1)
        # 2. 使用 FigureCanvas 将绘图渲染为 NumPy 数组
        canvas = FigureCanvas(fig)
        canvas.draw()

        # 3. 获取渲染后的图像数据（RGB）
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))  # 转换为 (height, width, 3)

        # 4. 关闭 Matplotlib 图以节省内存
        plt.close(fig)
        return img

    def close(self):
        return

def create_123bus(injection_bus):
    pp_net = pp.converter.from_mpc('F:/xuance/myCode/pandapower models/case_123.mat', casename_mpc_file='case_mpc')
    for bus in injection_bus:
        pp.create_sgen(pp_net, bus, p_mw=0, q_mvar=0)

    # 电网模型信息
    voltage_levels = pp_net.bus['vn_kv'].unique()  # 获取所有不同的电压等级
    print("电压等级（kV）:", voltage_levels)
    base_capacity = pp_net.sn_mva  # 网络的基准容量（标幺基准容量）
    print("基准容量（MVA）:", base_capacity)
    # print("\n节点信息：")
    # print(pp_net.bus[['name', 'vn_kv', 'zone', 'type']])

    return pp_net

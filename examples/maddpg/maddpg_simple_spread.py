import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import MADDPG_Agents


# 解析命令行参数的函数
def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: MADDPG for MPE.")
    parser.add_argument("--env-id", type=str, default="simple_spread_v3")  # 环境 ID
    parser.add_argument("--test", type=int, default=0)  # 测试模式标志
    parser.add_argument("--benchmark", type=int, default=1)  # 基准测试模式标志

    return parser.parse_args()


# 主程序入口
if __name__ == "__main__":
    # 解析命令行参数
    parser = parse_args()
    # 从配置文件中读取参数
    configs_dict = get_configs(file_dir=f"maddpg_mpe_configs/{parser.env_id}.yaml")
    # 更新配置字典，合并命令行参数
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    # 设置随机种子
    set_seed(configs.seed)
    # 创建环境实例
    envs = make_envs(configs)
    # 初始化 MADDPG 多智能体对象
    Agent = MADDPG_Agents(config=configs, envs=envs)

    # 显示训练信息
    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():
        print(f"{k}: {v}")

    # 如果启用基准测试模式
    if configs.benchmark:
        # 定义测试环境生成函数
        def env_fn():
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            return make_envs(configs_test)


        # 计算训练步数、评估间隔和总训练周期数
        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        # 测试智能体并保存最佳模型
        test_scores = Agent.test(env_fn, test_episode)
        Agent.save_model(model_name="best_model.pth")
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": Agent.current_step}

        # 训练过程循环
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agent.train(eval_interval)  # 训练指定的步数
            test_scores = Agent.test(env_fn, test_episode)  # 测试当前模型

            # 如果当前模型表现更好，保存为最佳模型
            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agent.current_step}
                # 保存最佳模型
                Agent.save_model(model_name="best_model.pth")

        # 输出最佳模型评分信息
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))

    # 测试模式
    else:
        if configs.test:
            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)


            # 加载模型并进行测试
            Agent.load_model(path=Agent.model_dir_load)
            scores = Agent.test(env_fn, configs.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")

        # 训练模式
        else:
            Agent.train(configs.running_steps // configs.parallels)  # 训练到指定步数
            Agent.save_model("final_train_model.pth")  # 保存最终模型
            print("Finish training!")

    # 结束智能体的所有操作
    Agent.finish()

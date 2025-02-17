import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from xuance.environment import make_envs
from xuance import get_configs
import argparse
from xuance.torch.agents import MATD3_Agents


if __name__ == "__main__":
    config_dict = get_configs(file_dir="Configs/MATD3_configs.yaml")
    configs = argparse.Namespace(**config_dict)
    '''
    在主文件中进行参数配置，用于kaggle
    '''
    configs.start_training = 0
    configs.running_steps = 10000
    configs.test = False

    envs = make_envs(configs)
    Agent = MATD3_Agents(config=configs, envs=envs)
    if configs.test:
        def env_fn():
            configs.parallels = configs.test_episode
            return make_envs(configs)
        Agent.load_model(path=Agent.model_dir_load)
        scores = Agent.test(env_fn, configs.test_episode)
        print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
        print("Finish testing.")
    else:
        Agent.train(configs.running_steps // configs.parallels)
        Agent.save_model("final_train_model.pth")
        print("Finish training!")
    Agent.finish()  # Finish the training.
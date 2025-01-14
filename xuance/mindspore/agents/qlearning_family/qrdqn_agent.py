from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.mindspore import Module
from xuance.mindspore.utils import NormalizeFunctions, ActivationFunctions, InitializeFunctions
from xuance.mindspore.policies import REGISTRY_Policy
from xuance.mindspore.agents.qlearning_family.dqn_agent import DQN_Agent


class QRDQN_Agent(DQN_Agent):
    """The implementation of QRDQN agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(QRDQN_Agent, self).__init__(config, envs)

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = InitializeFunctions[self.config.initialize] if hasattr(self.config, "initialize") else None
        activation = ActivationFunctions[self.config.activation]

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "QR_Q_network":
            policy = REGISTRY_Policy["QR_Q_network"](
                action_space=self.action_space, quantile_num=self.config.quantile_num,
                representation=representation, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation)
        else:
            raise AttributeError(f"{self.config.agent} currently does not support the policy named {self.config.policy}.")

        return policy

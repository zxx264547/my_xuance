from xuance.environment.utils import EnvironmentDict
from xuance.common import Optional
from xuance.environment.powerSystem.IEEE13 import IEEE13

from xuance.environment import REGISTRY_MULTI_AGENT_ENV
REGISTRY_MULTI_AGENT_ENV["IEEE13"] = IEEE13


# 注册自定义环境
try:
    from xuance.environment.powerSystem.IEEE13 import IEEE13
    REGISTRY_MULTI_AGENT_ENV['IEEE13'] = IEEE13
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["IEEE13"] = str(error)

from xuance.environment.utils import EnvironmentDict
from xuance.common import Optional
from xuance.environment.powerSystem.IEEE123 import IEEE123

from xuance.environment import REGISTRY_MULTI_AGENT_ENV
REGISTRY_MULTI_AGENT_ENV["IEEE123"] = IEEE123


# 注册自定义环境
try:
    from xuance.environment.powerSystem.IEEE123 import IEEE123
    REGISTRY_MULTI_AGENT_ENV['IEEE13'] = IEEE123
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["IEEE13"] = str(error)

# create enum of policy
from enum import Enum
class Policy(Enum):
    PPO = "PPO"
    A2C = "A2C"
    DQN = "DQN"



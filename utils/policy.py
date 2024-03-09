from enum import Enum, auto

class Policy(Enum):
    PPO = auto()
    A2C = auto()
    DQN = auto()
    Random = auto()
    Deterministic = auto()
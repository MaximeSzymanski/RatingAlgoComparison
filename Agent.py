import numpy as np
from models.PPO import PPO
from pettingzoo.utils.env import AECEnv
class Agent():
    def __init__(self,policy_type,state_size, action_size) -> None:
        if policy_type == "PPO":
            self.policy = PPO(state_size=state_size, action_size=action_size,num_steps=40048,
                              batch_size=32)
        else:
            raise ValueError("Policy not found")


        self.rating = 0




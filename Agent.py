import numpy as np
from models.PPO import PPO
from models.DQN import DQN
from models.A2C import A2C
from models.Random import Random
from models.Deterministic import Deterministic
from utils.policy import Policy

from pettingzoo.utils.env import AECEnv

class Agent():
    def __init__(self,policy_type : Policy,state_size, action_size,id,seed=0,action_deterministic = 0) -> None:
        if policy_type == Policy.PPO:
            self.policy = PPO(state_size=state_size, action_size=action_size,num_steps=2048,
                              batch_size=32,env_name='connect_four_v3')
        elif policy_type == Policy.DQN:
            self.policy= DQN(memory_size=10000,batch_size=64,gamma=0.99,epsilon=1,epsilon_decay=0.95,epsilon_min=0.01,
                             state_size=state_size,action_size=action_size,seed=42,env_name="connect_four_v3",lr=3e-4)

        elif policy_type == Policy.A2C:
            self.policy = A2C(state_size=state_size, action_size=action_size, num_steps=5,env_name='connect_four_v3')
        elif policy_type == Policy.Random:
            self.policy = Random(action_size=action_size,seed=seed)

        elif policy_type == Policy.Deterministic:
            self.policy = Deterministic(action_size=action_size,action_index=action_deterministic)

        else:
            raise ValueError("Policy not found")
        self.policy_type = policy_type
        self.policy_name = str(policy_type) + (f" {action_deterministic}" if policy_type == Policy.Deterministic else "")
        # remove "Policy." from the policy name
        self.policy_name = self.policy_name.split(".")[1]
        self.state_size = state_size
        self.action_size = action_size
        self.id = id



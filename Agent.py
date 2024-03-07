import numpy as np
from models.PPO import PPO
from models.DQN import DQN
from utils.policy import Policy
from pettingzoo.utils.env import AECEnv

class Agent():
    def __init__(self,policy_type : Policy,state_size, action_size,id) -> None:
        if policy_type == Policy.PPO:
            self.policy = PPO(state_size=state_size, action_size=action_size,num_steps=2048,
                              batch_size=32,env_name='connect_four_v3')
        elif policy_type == Policy.DQN:
            self.policy= DQN(memory_size=10000,batch_size=64,gamma=0.99,epsilon=1,epsilon_decay=0.95,epsilon_min=0.01,
                             state_size=state_size,action_size=action_size,seed=42,env_name="connect_four_v3",lr=3e-4)
        else:
            raise ValueError("Policy not found")
        self.policy_type = policy_type
        self.state_size = state_size
        self.action_size = action_size
        self.id = id



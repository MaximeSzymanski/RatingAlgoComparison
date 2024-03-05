import numpy as np

class Agent():

    def __init__(self,policy_type,env) -> None:
        if policy_type == "PPO":
            self.policy = PPOTrainer(env)



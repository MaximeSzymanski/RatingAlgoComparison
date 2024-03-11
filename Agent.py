import numpy as np
from models.PPO import PPO
from models.DQN import DQN
from models.A2C import A2C
from models.Random import Random
from models.Deterministic import Deterministic
from utils.policy import Policy

from pettingzoo.utils.env import AECEnv


class Agent():
    def __init__(
            self,
            policy_type: Policy,
            state_size: int,
            action_size: int,
            id: int,
            agent_index: int,
            action_deterministic: int = 0,
            seed: int = 0
    ) -> None:
        """
        Initialize an agent.

        Parameters:
            policy_type (Policy): The type of policy for the agent.
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            id (int): The identifier of the agent.
            action_deterministic (int): The deterministic action for Deterministic policy.
            agent_index (int): The index of the agent.
            seed (int, optional): The seed for random number generation (default is 0).
        """
        # Initialize policy based on the specified policy type
        if policy_type == Policy.PPO:
            self.policy = PPO(state_size=state_size, action_size=action_size, num_steps=2048,
                              batch_size=64, env_name='connect_four_v3', agent_index=agent_index)
        elif policy_type == Policy.DQN:
            self.policy = DQN(memory_size=10000, batch_size=64, gamma=0.99, epsilon=1, epsilon_decay=0.98,
                              epsilon_min=0.01, state_size=state_size, action_size=action_size, seed=42,
                              env_name="connect_four_v3", lr=3e-4, agent_index=agent_index)
        elif policy_type == Policy.A2C:
            self.policy = A2C(state_size=state_size, action_size=action_size, num_steps=5, env_name='connect_four_v3',
                              agent_index=agent_index)
        elif policy_type == Policy.Random:
            self.policy = Random(action_size=action_size, seed=seed)
        elif policy_type == Policy.Deterministic:
            self.policy = Deterministic(
                action_size=action_size, action_index=action_deterministic)
        else:
            raise ValueError("Policy not found")

        # Set attributes
        self.policy_type = policy_type
        self.policy_name = str(policy_type) + (
            f" {action_deterministic}" if policy_type == Policy.Deterministic else "")
        # Remove "Policy." from the policy name
        self.policy_name = self.policy_name.split(".")[1]
        self.state_size = state_size
        self.action_size = action_size
        self.id = id
        self.num_fights = 0

    def get_action_distribution(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Get the action distribution of the agent.

        Parameters:
            state (np.ndarray): The state of the environment.
            mask (np.ndarray): The mask for valid actions.

        Returns:
            np.ndarray: The action distribution of the agent.
        """
        action_distribution = self.policy.get_action_distribution(state, mask)
        assert np.allclose(np.sum(action_distribution),
                           1), f"Action distribution of agent {self.id} does not sum to 1"
        return action_distribution

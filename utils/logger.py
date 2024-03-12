import numpy as np
from utils.plot import *
import os
from Agent import Agent
from utils.policy import Policy

class Logger():
    def __init__(self, log_file):
        os.makedirs("logs", exist_ok=True)

        pass

    def log_diversity_matrix(self, diversity_matrix : np.array, num_trial : int, num_round : int, agents : list[Agent]) -> None:
        """
        Log the diversity matrix.
        """
        os.makedirs("logs/diversity_matrix", exist_ok=True)
        os.makedirs(f"logs/diversity_matrix/{num_trial}", exist_ok=True)
        path = f"logs/diversity_matrix/{num_trial}"
        plot_diversity_matrix(diversity_matrix=diversity_matrix, index_file=num_round, path=path, agents = agents)

    def log_diversity_per_policy_trial_until_round(self, diversity_per_agent : dict[Policy, np.array], num_trial : int, num_round : int) -> None:
        """
        Log the diversity per policy.
        """
        os.makedirs("logs/diversity_per_policy", exist_ok=True)
        os.makedirs(f"logs/diversity_per_policy/{num_trial}", exist_ok=True)
        path = f"logs/diversity_per_policy/{num_trial}"
        plot_diversity_per_policy_round(diversity_per_policy=diversity_per_agent, index_file=num_round, path=path)

    def log_diversity_per_type_of_policy_averaged_over_trials(self, diversity_per_type : dict[str, tuple[np.array,np.array]]) -> None:
        """
        Log the diversity per type of policy.
        """
        os.makedirs("logs/diversity_per_type", exist_ok=True)
        path = "logs/diversity_per_policy"
        plot_diversity_per_type_of_policy_averaged_over_trials(diversity_per_type=diversity_per_type, path=path)

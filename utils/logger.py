import numpy as np
from utils.plot import *
import os
from Agent import Agent
from utils.policy import Policy
from rating.rating import RatingSystem

class Logger():
    def __init__(self, log_file):
        os.makedirs("logs", exist_ok=True)

        pass

    def plot_rating_distribution(self,  num_trial : int, num_round : int, rating : RatingSystem) -> None:
        """
        Plot the rating distribution.
        Params:
            rating_distribution (np.array): The rating distribution.
        """
        os.makedirs(f"logs/rating_distribution_{rating.name}/{num_trial}", exist_ok=True)
        path = f"logs/rating_distribution_{rating.name}/{num_trial}"
        rating.plot_distribution(path=path,round= num_round)
    def plot_rating_per_policy(self, policies: List[str], rating_mean: Dict[str, List[int]],
                               rating_std: Dict[str, List[int]],num_trial : int,rating : RatingSystem) -> None:
        """
        Plot the rating per policy.
        Params:
            policies (List[str]): The list of policies.
            rating_mean (Dict[str, List[int]]): The mean rating per policy.
            rating_std (Dict[str, List[int]]): The std rating per policy.
        """

        os.makedirs(f"logs/rating_per_policy_{rating.name}", exist_ok=True)
        path = f"logs/rating_per_policy_{rating.name}"
        rating.plot_rating_per_policy(policies=policies, rating_mean=rating_mean, rating_std=rating_std, path=path,num_trial=num_trial)




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

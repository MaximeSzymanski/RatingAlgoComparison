import numpy as np
from utils.plot import *
import os
from Agent import Agent
from utils.policy import Policy
from poprank import Rate
import matplotlib.pyplot as plt
#from rating.rating import RatingSystem

class Logger():
    def __init__(self, verbose: bool = False) -> None:
        os.makedirs("logs", exist_ok=True)
        self.verbose = verbose

        pass

    def plot_rating_distribution(self,  num_trial : int, num_round : int, ratings : "List[Rate]", rating_name : str, experiment : str) -> None:
        """
        Plot the rating distribution.
        Params:
            rating_distribution (np.array): The rating distribution.
        """

        os.makedirs(f"logs/{experiment}/rating_distribution_{rating_name}/{num_trial}", exist_ok=True)
        path = f"logs/{experiment}/rating_distribution_{rating_name}/{num_trial}"

        ratings = [r.mu for r in ratings]
        plt.hist(ratings, bins=20, edgecolor='black')
        plt.xlabel(rating_name)
        plt.ylabel('Frequency')
        plt.title(f'{rating_name} Distribution')
        # save into file
        plt.savefig(f"{path}/{rating_name}_distribution_{num_round}.png")
        plt.clf()

    def plot_rating_per_policy(self, policies: List[str], rating_mean: Dict[str, List[int]],
                               rating_std: Dict[str, List[int]],num_trial : int,rating : Rate, experiment : str) -> None:
        """
        Plot the rating per policy.
        Params:
            policies (List[str]): The list of policies.
            rating_mean (Dict[str, List[int]]): The mean rating per policy.
            rating_std (Dict[str, List[int]]): The std rating per policy.
        """
        if self.verbose:
            print("Plotting rating per policy...")
            print(f"Best policy: {max(rating_mean, key=rating_mean.get)}")


        os.makedirs(f"logs/{experiment}/rating_per_policy_{rating.name}", exist_ok=True)
        path = f"logs/{experiment}/rating_per_policy_{rating.name}"
        rating.plot_rating_per_policy(policies=policies, rating_mean=rating_mean, rating_std=rating_std, path=path,num_trial=num_trial)

    def log_diversity_matrix(self, diversity_matrix : np.array, num_trial : int, num_round : int, agents : list[Agent], experiment : str) -> None:
        """
        Log the diversity matrix.
        """
        os.makedirs(f"logs/{experiment}/diversity_matrix", exist_ok=True)
        os.makedirs(f"logs/{experiment}/diversity_matrix/{num_trial}", exist_ok=True)
        path = f"logs/{experiment}/diversity_matrix/{num_trial}"
        plot_diversity_matrix(diversity_matrix=diversity_matrix, index_file=num_round, path=path, agents = agents)

    def log_diversity_per_policy_trial_until_round(self, diversity_per_agent : dict[Policy, np.array], num_trial : int, num_round : int, experiment : str) -> None:
        """
        Log the diversity per policy.
        """
        os.makedirs(f"logs/{experiment}/diversity_per_policy", exist_ok=True)
        os.makedirs(f"logs/{experiment}/diversity_per_policy/{num_trial}", exist_ok=True)
        path = f"logs/{experiment}/diversity_per_policy/{num_trial}"
        plot_diversity_per_policy_round(diversity_per_policy=diversity_per_agent, index_file=num_round, path=path)

    def log_diversity_per_type_of_policy_averaged_over_trials(self, diversity_per_type : dict[str, tuple[np.array,np.array]], experiment: str) -> None:
        """
        Log the diversity per type of policy.
        """
        os.makedirs(f"logs/{experiment}/diversity_per_type", exist_ok=True)
        path = f"logs/{experiment}/diversity_per_policy"
        plot_diversity_per_type_of_policy_averaged_over_trials(diversity_per_type=diversity_per_type, path=path)
    
    def plot_agents_rating_over_time(self, ratings_over_time, rating_name: str, experiment: str):
        ratings_over_time = np.array(ratings_over_time)
        n_timesteps, n_agents = ratings_over_time.shape
        time_steps = np.arange(n_timesteps)
        
        #plt.figure(figsize=(10, 6))  # Adjust figure size as needed
        os.makedirs(f"logs/{experiment}/rating_over_time", exist_ok=True)
        path = f"logs/{experiment}/rating_over_time/{rating_name}_over_time.png"
        
        for agent_idx in range(n_agents):
            plt.plot(time_steps, ratings_over_time[:, agent_idx], label=f"Agent {agent_idx+1}")

        plt.title('Agent Ratings Over Time')
        plt.xlabel('Time')
        plt.ylabel('Rating')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(path)
        plt.clf()

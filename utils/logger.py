import os
import matplotlib.pyplot as plt
import numpy as np
from Agent import Agent
from utils.plot import plot_and_save_diversity_matrix, plot_and_save_diversity_over_time_global, plot_and_save_diversity_over_time_per_policy_type
from utils.policy import Policy

class Logger:
    def __init__(self):
        """
        Initialize Logger object.
        """
        self.diversity_matrix_log_counter = 0

    def log_diversity_matrix(self, diversity_matrix: np.array, agents: list[Agent]) -> None:
        """
        Log the diversity matrix.

        Args:
            diversity_matrix (np.array): The diversity matrix.
            agents (list[Agent]): The list of agents.
        """
        # create the folder if it does not exist
        os.makedirs("diversity_matrix", exist_ok=True)
        plot_and_save_diversity_matrix(diversity_matrix, agents, self.diversity_matrix_log_counter)
        self.diversity_matrix_log_counter += 1

    def log_diversity_over_time_global(self, diversity_over_time: np.array, number_round: int) -> None:
        """
        Log the diversity over time.

        Args:
            diversity_over_time (np.array): The diversity over time.
            number_round (int): The number of rounds to plot (optional).
        """
        # create the folder if it does not exist
        os.makedirs("diversity_score_global", exist_ok=True)
        plot_and_save_diversity_over_time_global(diversity_over_time, number_round)

    def log_diversity_over_time_per_policy_type(self, diversity_over_time: dict[Policy, np.array], number_round: int) -> None:
        """
        Log the diversity over time for each policy type.

        Args:
            diversity_over_time (dict[Policy, np.array]): The diversity over time for each policy type.
            number_round (int): The number of rounds to plot (optional).
        """
        # create the folder if it does not exist
        os.makedirs("diversity_score_per_agent", exist_ok=True)
        plot_and_save_diversity_over_time_per_policy_type(diversity_over_time=diversity_over_time, number_round=number_round)

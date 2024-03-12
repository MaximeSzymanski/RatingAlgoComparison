import os
import matplotlib.pyplot as plt
import numpy as np
from Agent import Agent
from utils.plot import *
from utils.policy import Policy


class Logger:
    def __init__(self):
        """
        Initialize Logger object.
        """
        self.diversity_matrix_log_counter = 0

    def log_diversity_matrix(self, diversity_matrix: np.array, agents: list[Agent],num_trials : int ,num_round : int ) -> None:
        """
        Log the diversity matrix. ( upper triangular matrix )

        Args:
            diversity_matrix (np.array): The diversity matrix.
            agents (list[Agent]): The list of agents.
            num_trials (int): The number of trials.
        """
        # create the folder if it does not exist
        os.makedirs("diversity_matrix", exist_ok=True)
        os.makedirs(f"diversity_matrix/{num_trials}", exist_ok=True)
        if len(diversity_matrix.shape) == 2:
            diversity_matrix = diversity_matrix.reshape(1, *diversity_matrix.shape)
        # keep only upper triangular matrix

        plot_and_save_diversity_matrix(
            diversity_matrix, agents, num_round,num_trial=num_trials)
        self.diversity_matrix_log_counter =  (self.diversity_matrix_log_counter + 1) % diversity_matrix.shape[0]

    def log_diversity_matrix_all_trials(self, diversity_matrix: np.array, agents: list[Agent] ) -> None:
        """
        Log the diversity matrix. ( upper triangular matrix )

        Args:
            diversity_matrix (np.array): The diversity matrix.
            agents (list[Agent]): The list of agents.
            num_trials (int): The number of trials.
        """
        # create the folder if it does not exist
        os.makedirs("diversity_matrix", exist_ok=True)
        os.makedirs(f"diversity_matrix/all_trials/", exist_ok=True)

        # keep only upper triangular matrix

        plot_and_save_diversity_matrix_whole_trials(
            diversity_matrix, agents)
        self.diversity_matrix_log_counter =  (self.diversity_matrix_log_counter + 1) % diversity_matrix.shape[0]



    def log_diversity_over_time_global(self, diversity_over_time: np.array, number_round: int,num_trials : int) -> None:
        """
        Log the diversity over time.

        Args:
            diversity_over_time (np.array): The diversity over time.
            number_round (int): The number of rounds to plot (optional).
        """
        # create the folder if it does not exist
        os.makedirs("diversity_score_global", exist_ok=True)
        os.makedirs(f"diversity_score_global/{num_trials}", exist_ok=True)
        plot_and_save_diversity_over_time_global(
            diversity_over_time, number_round,num_trials=num_trials)

    def log_diversity_over_time_global_all_trials(self, diversity_over_time: np.array) -> None:
        """
        Log the diversity over time.

        Args:
            diversity_over_time (np.array): The diversity over time.
            number_round (int): The number of rounds to plot (optional).
        """
        # create the folder if it does not exist
        os.makedirs("diversity_score_global", exist_ok=True)
        os.makedirs(f"diversity_score_global/all_trials", exist_ok=True)
        plot_and_save_diversity_over_time_global_all_trials(
            diversity_over_time)
    def log_diversity_over_time_per_policy_type(self, diversity_over_time: dict[Policy, np.array], number_round: int,num_trials : int) -> None:
        """
        Log the diversity over time for each policy type.

        Args:
            diversity_over_time (dict[Policy, np.array]): The diversity over time for each policy type.
            number_round (int): The number of rounds to plot (optional).
        """
        # create the folder if it does not exist
        os.makedirs("diversity_score_per_agent", exist_ok=True)
        os.makedirs(f"diversity_score_per_agent/{num_trials}", exist_ok=True)
        plot_and_save_diversity_over_time_per_policy_type(
            diversity_over_time=diversity_over_time, number_round=number_round, num_trial=num_trials)

    def log_diversity_over_time_per_policy_type_all_trials(self, diversity_over_time: dict[Policy, np.array]) -> None:
        """
        Log the diversity over time for each policy type.

        Args:
            diversity_over_time (dict[Policy, np.array]): The diversity over time for each policy type.
            number_round (int): The number of rounds to plot (optional).
        """
        # create the folder if it does not exist
        os.makedirs("diversity_score_per_agent", exist_ok=True)
        os.makedirs(f"diversity_score_per_agent/all_trials", exist_ok=True)
        plot_and_save_diversity_over_time_per_policy_type_all_trials(
            diversity_over_time=diversity_over_time)
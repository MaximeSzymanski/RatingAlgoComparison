import os, matplotlib.pyplot as plt
import numpy as np
from Agent import Agent
from plot import plot_and_save_diversity_matrix

class Logger():
    def __init__(self):
        self.diversity_matrix_log_counter = 0


    def log_diversity_matrix(self, diversity_matrix : np.array, agents : list[Agent]):
        """
        Log the diversity matrix
        :param diversity_matrix: The diversity matrix
        """
        # create the folder if it does not exist
        os.makedirs("diversity_matrix", exist_ok=True)
        plot_and_save_diversity_matrix(diversity_matrix,agents, self.diversity_matrix_log_counter)
        self.diversity_matrix_log_counter += 1




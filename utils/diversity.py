from scipy.stats import wasserstein_distance
import numpy as np
from typing import Dict
class Diversity():
    def __init__(self,number_agents : int):
        """
        Class to compute the diversity of a population
        :param number_agents: The number of agents in the population
        """
        self.distance_matrix = np.zeros((number_agents, number_agents))



    def compute_distance_between_two_states(self, state1 : np.ndarray, state2 : np.ndarray) -> float:
        """
        Compute the distance between two states
        :param state1: The first state
        :param state2: The second state
        :return: The distance between the two states
        """
        return wasserstein_distance(u_values=state1, v_values=state2)

    def compute_distance_between_two_trajectories(self, trajectory1 : np.ndarray, trajectory2 : np.ndarray) -> float:
        """
        Compute the distance between two trajectories
        :param trajectory1: The first trajectory
        :param trajectory2: The second trajectory
        :return: The distance between the two trajectories
        """
        return wasserstein_distance(u_values=trajectory1, v_values=trajectory2)

    def compute_distance_matrix(self, agents_data : Dict[int, list[np.array]]) -> np.ndarray:
        """
        Compute the distance matrix
        :param agents_data: The data of the agents. It is a dictionary with the agent id as key and the data as value. The value
        is a list of numpy arrays. Each numpy array is a state of the agent.
        :return: The distance matrix of the agents
        """
        for i in agents_data.keys():
            for j in agents_data.keys():
                self.distance_matrix[i][j] = self.compute_distance_between_two_trajectories(agents_data[i], agents_data[j])
        return self.distance_matrix

    def compute_diversity(self, agents_data : Dict[int, list[np.array]]) -> float:
        """
        Compute the diversity of the population
        :param agents_data: The data of the agents. It is a dictionary with the agent id as key and the data as value. The value
        is a list of numpy arrays. Each numpy array is a state of the agent.
        :return: The diversity of the population
        """
        self.compute_distance_matrix(agents_data)
        return np.mean(self.distance_matrix)


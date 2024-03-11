
from Agent import Agent
import numpy as np
# import scaler minmax
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt
from functools import cache
class DiversityAction():

    def __init__(self,number_agent : int = 0,number_round=1) -> None:
        self.distance_matrix = np.zeros((number_agent, number_agent))
        self.distance_score = np.zeros((number_round, number_agent))
        self.current_round = 0
    def get_diversity_two_agents(self,agent1 : Agent, agent2 : Agent, list_states : list[np.array], list_masks : list[np.array]) -> float:
        """
        Get the diversity between two agents. This is the cross entropy between the action distribution of the two agents.
        :param agent1: The first agent
        :param agent2: The second agent
        :return: The diversity between the two agents
        """

        diversity = 0
        for state, mask in zip(list_states, list_masks):
            action_distribution_agent1 = agent1.get_action_distribution(state, mask)
            action_distribution_agent2 = agent2.get_action_distribution(state, mask)
            diversity += self.cross_entropy(action_distribution_agent1, action_distribution_agent2) + self.cross_entropy(action_distribution_agent2, action_distribution_agent1)

        # normalize the diversity
        diversity /= len(list_states)
        return diversity

    def cross_entropy(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute the cross entropy between two distributions
        :param p: First distribution
        :param q: Second distribution
        :return: Cross entropy
        """
        # Make sure p and q have the same shape
        assert p.shape == q.shape, "Distributions must have the same shape"
        cross_entropy = -np.sum(p * np.log(q+1e-10))
        return cross_entropy

    def KL_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute the KL divergence between two distributions
        :param p: First distribution
        :param q: Second distribution
        :return: KL divergence
        """
        # Make sure p and q have the same shape
        assert p.shape == q.shape, "Distributions must have the same shape"

        # Compute the KL divergence with a small epsilon added to avoid division by zero
        kl_divergence = np.sum(p * np.nan_to_num(np.log((p / (q + 1e-8)))))


        return kl_divergence




    def compute_diversity(self, agents : list[Agent], list_states : list[np.array], list_masks : list[np.array]) -> np.array:
        """
        Compute the diversity of the population
        :param agents: The agents
        :param list_states: The list of states
        :param list_masks: The list of masks
        :return: The diversity matrix of the agents
        """
        diversity = 0
        for i in range(len(agents)):
            for j in range(len(agents)):
                diversity = self.get_diversity_two_agents(agents[i], agents[j], list_states, list_masks)
                self.distance_matrix[i, j] = diversity / len(list_states)
        self.update_distance_score()
        return self.distance_matrix

    def update_distance_score(self) -> None:
        """
        Compute the distance score
        :return: The distance score
        """
        # sum the lines of the distance matrix
        self.distance_score[self.current_round] = np.sum(self.distance_matrix, axis=1)
        self.current_round += 1
        print(f"shape of distance score: {self.distance_score.shape}")



    def get_distance_score(self) -> np.array:
        """
        Get the distance score
        :return: The distance score
        """
        return self.distance_score
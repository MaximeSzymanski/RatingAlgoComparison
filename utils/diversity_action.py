
from Agent import Agent
import numpy as np
# import scaler minmax
from sklearn.preprocessing import MinMaxScaler
import torch
from utils.policy import Policy
import matplotlib.pyplot as plt
from functools import cache
class DiversityAction():

    def __init__(self,number_agent : int = 0,number_round=1,non_random_deterministic_agent : int = 0 ,id_agent_to_policy : dict[int, Policy] = {}) -> None:
        self.total_distance_matrix = np.zeros((number_agent, number_agent))
        self.non_random_deterministic_agent = non_random_deterministic_agent
        self.distance_score = np.zeros((number_round, self.non_random_deterministic_agent))
        self.policy_per_id_agent = id_agent_to_policy


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

        # check if self.non_random_deterministic_agent is respected : the first self.non_random_deterministic_agent agents are non-random and non-deterministic
        assert len(agents) >= self.non_random_deterministic_agent, "The number of non-random and non-deterministic agents is greater than the number of agents"
        if self.non_random_deterministic_agent > 0:
            for i in range(self.non_random_deterministic_agent):
                assert not agents[i].policy_type == Policy.Random or Policy.Deterministic , "The first agents must be non-random and non-deterministic"

        diversity = 0
        for i in range(len(agents)):
            for j in range(len(agents)):
                diversity = self.get_diversity_two_agents(agents[i], agents[j], list_states, list_masks)
                self.total_distance_matrix[i, j] = diversity / len(list_states)

        self.update_distance_score()
        return self.total_distance_matrix

    def update_distance_score(self) -> None:
        """
        Compute the distance score only for the non-random and non-deterministic agents
        :return: The distance score
        """
        # sum the lines of the distance matrix

        self.distance_score[self.current_round] = np.sum(self.total_distance_matrix[:self.non_random_deterministic_agent], axis=1)
        self.current_round += 1

    def get_distance_score_per_policy_type(self) -> dict[Policy, np.array]:
        """
        Get the distance score per policy type
        :return: The distance score per policy type
        """
        # Filter out 'Random' and 'Deterministic' policies
        policy_to_plot = [policy for policy in Policy if policy not in [Policy.Random, Policy.Deterministic]]

        # Initialize distance score per policy type dictionary
        distance_score_per_policy_type = {
            policy: np.zeros((self.distance_score.shape[0],
                              sum(1 for agent_policy in self.policy_per_id_agent.values() if agent_policy == policy)))
            for policy in policy_to_plot
        }

        # Keep track of current agent index for each policy type
        current_agent_index_for_current_policy = {policy: 0 for policy in policy_to_plot}

        # Fill distance score per policy type
        for agent, index in enumerate(self.policy_per_id_agent.keys()):
            policy_type = self.policy_per_id_agent[agent]
            if policy_type in policy_to_plot:
                distance_score_per_policy_type[policy_type][:,
                current_agent_index_for_current_policy[policy_type]] = self.distance_score[:, agent]
                current_agent_index_for_current_policy[policy_type] += 1

        # Normalize the distance score by dividing by the number of agents of the same policy type
        for policy in policy_to_plot:
            agents_count = sum(1 for agent_policy in self.policy_per_id_agent.values() if agent_policy == policy)
            distance_score_per_policy_type[policy] /= agents_count
            if agents_count == 1:
                distance_score_per_policy_type[policy] = distance_score_per_policy_type[policy].reshape(-1, 1)

        return distance_score_per_policy_type

    def get_distance_score_global(self) -> np.array:
        """
        Get the distance score
        :return: The distance score
        """
        return self.distance_score
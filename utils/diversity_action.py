from Agent import Agent
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from utils.policy import Policy
import matplotlib.pyplot as plt
from functools import cache


class DiversityAction:

    def __init__(self, number_agent: int , number_round: int , non_random_deterministic_agent: int ,
                 id_agent_to_policy: dict[int, Policy]) -> None:
        """
        Initialize DiversityAction object.

        Args:
            number_agent (int): The number of agents.
            number_round (int): The number of rounds.
            non_random_deterministic_agent (int): The number of non-random and non-deterministic agents.
            id_agent_to_policy (dict[int, Policy]): A dictionary mapping agent IDs to policy types.
        """
        self.total_distance_matrix = np.zeros((number_agent, number_agent))
        self.non_random_deterministic_agent = non_random_deterministic_agent
        self.distance_score = np.zeros((number_round, self.non_random_deterministic_agent))
        self.policy_per_id_agent = id_agent_to_policy
        self.current_round = 0

    def get_diversity_two_agents(self, agent1: Agent, agent2: Agent, list_states: list[np.array],
                                 list_masks: list[np.array]) -> float:
        """
        Get the diversity between two agents.

        Args:
            agent1 (Agent): The first agent.
            agent2 (Agent): The second agent.
            list_states (list[np.array]): The list of states.
            list_masks (list[np.array]): The list of masks.

        Returns:
            float: The diversity between the two agents.
        """
        diversity = 0
        for state, mask in zip(list_states, list_masks):
            action_distribution_agent1 = agent1.get_action_distribution(state, mask)
            action_distribution_agent2 = agent2.get_action_distribution(state, mask)
            diversity += self.cross_entropy(action_distribution_agent1,
                                            action_distribution_agent2) + self.cross_entropy(action_distribution_agent2,
                                                                                             action_distribution_agent1)

        # normalize the diversity
        diversity /= len(list_states)
        return diversity

    def cross_entropy(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute the cross entropy between two distributions.

        Args:
            p (np.ndarray): First distribution.
            q (np.ndarray): Second distribution.

        Returns:
            float: Cross entropy.
        """
        assert p.shape == q.shape, "Distributions must have the same shape"
        cross_entropy = -np.sum(p * np.log(q + 1e-10))
        return cross_entropy

    def KL_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute the KL divergence between two distributions.

        Args:
            p (np.ndarray): First distribution.
            q (np.ndarray): Second distribution.

        Returns:
            float: KL divergence.
        """
        assert p.shape == q.shape, "Distributions must have the same shape"
        kl_divergence = np.sum(p * np.nan_to_num(np.log((p / (q + 1e-8)))))
        return kl_divergence

    def compute_diversity(self, agents: list[Agent], list_states: list[np.array],
                          list_masks: list[np.array]) -> np.array:
        """
        Compute the diversity of the population.

        Args:
            agents (list[Agent]): The agents.
            list_states (list[np.array]): The list of states.
            list_masks (list[np.array]): The list of masks.

        Returns:
            np.array: The diversity matrix of the agents.
        """
        assert len(
            agents) >= self.non_random_deterministic_agent, "The number of non-random and non-deterministic agents is greater than the number of agents"
        if self.non_random_deterministic_agent > 0:
            for i in range(self.non_random_deterministic_agent):
                assert not agents[
                               i].policy_type == Policy.Random or Policy.Deterministic, "The first agents must be non-random and non-deterministic"

        diversity = 0
        for i in range(len(agents)):
            for j in range(len(agents)):
                diversity = self.get_diversity_two_agents(agents[i], agents[j], list_states, list_masks)
                self.total_distance_matrix[i, j] = diversity / len(list_states)

        self.update_distance_score()
        return self.total_distance_matrix

    def update_distance_score(self) -> None:
        """
        Compute the distance score only for the non-random and non-deterministic agents.
        """
        self.distance_score[self.current_round] = np.sum(
            self.total_distance_matrix[:self.non_random_deterministic_agent], axis=1)
        self.current_round += 1

    def get_distance_score_per_policy_type(self) -> dict[Policy, np.array]:
        """
        Get the distance score per policy type.

        Returns:
            dict[Policy, np.array]: The distance score per policy type.
        """
        policy_to_plot = [policy for policy in Policy if policy not in [Policy.Random, Policy.Deterministic]]
        distance_score_per_policy_type = {
            policy: np.zeros((self.distance_score.shape[0],
                              sum(1 for agent_policy in self.policy_per_id_agent.values() if agent_policy == policy)))
            for policy in policy_to_plot
        }
        current_agent_index_for_current_policy = {policy: 0 for policy in policy_to_plot}

        for agent, index in enumerate(self.policy_per_id_agent.keys()):
            policy_type = self.policy_per_id_agent[agent]
            if policy_type in policy_to_plot:
                distance_score_per_policy_type[policy_type][:,
                current_agent_index_for_current_policy[policy_type]] = self.distance_score[:, agent]
                current_agent_index_for_current_policy[policy_type] += 1

        for policy in policy_to_plot:
            agents_count = sum(1 for agent_policy in self.policy_per_id_agent.values() if agent_policy == policy)
            distance_score_per_policy_type[policy] /= agents_count
            if agents_count == 1:
                distance_score_per_policy_type[policy] = distance_score_per_policy_type[policy].reshape(-1, 1)

        return distance_score_per_policy_type

    def get_distance_score_global(self) -> np.array:
        """
        Get the distance score.

        Returns:
            np.array: The distance score.
        """
        return self.distance_score

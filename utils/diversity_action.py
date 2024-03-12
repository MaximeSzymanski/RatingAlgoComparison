import numpy as np
from Agent import Agent

class Diversity():

    """
    This class is used to calculate the diversity of the population of agents.
    It should contains the diversity for each pair of agents, for each round and for each trial.
    """

    def __init__(self, num_trials: int, num_rounds: int, num_agents: int, agents : list[Agent]) -> None:
        """
        Initialize the Diversity object.
        """
        self.diversity_matrix = np.zeros((num_trials, num_rounds, num_agents, num_agents))
        self.diversity_per_agent = {}
        for agent in agents:
            self.diversity_per_agent[agent.id] = {}
            for opponent in agents:
                self.diversity_per_agent[agent.id][opponent.id] =  np.zeros((num_trials, num_rounds))
        self.num_trials = num_trials
        self.num_rounds = num_rounds
        self.num_agents = num_agents
        self.agents = agents

    def calculate_diversity_pair(self, agent1: Agent, agent2: Agent, states : list[np.array], masks :  list[np.array]) -> float:
        """
        Calculate the diversity between two agents. Use cross-entropy for now.
        """
        diversity = 0
        for state,mask in zip(states, masks):
            action_distribution1 = agent1.policy.get_action_distribution(state, mask)
            action_distribution2 = agent2.policy.get_action_distribution(state, mask)
            diversity += self.cross_entropy(action_distribution1, action_distribution2) / len(states)
        return diversity


    def cross_entropy(self, p, q):
        """
        Compute the cross entropy between two distributions.
        """
        return -np.sum(p * np.log(q+1e-10))
    def calculate_diversity_round(self, num_trial: int, num_round: int, agents: list[Agent], states : np.array, masks : np.array) -> None:
        """
        Calculate the diversity of the population of agents for a specific round and trial.
        """
        for i, agent in enumerate(agents):
            for j, opponent in enumerate(agents):
                if i == j:
                    continue
                diversity_ij = self.calculate_diversity_pair(agent, opponent, states, masks)
                self.diversity_per_agent[agent.id][opponent.id][num_trial, num_round] = diversity_ij


    def build_diversity_matrix_round(self, num_trial: int, num_round: int) -> None:
        """
        Build the diversity matrix for a specific round and trial.
        """

        for id, record_dict in self.diversity_per_agent.items():
            for id2, record in record_dict.items():
                self.diversity_matrix[num_trial, num_round, id, id2] = record[num_trial, num_round]

    def get_diversity_matrix(self, num_trial: int, num_round: int) -> np.array:
        """
        Get the diversity matrix.
        """
        return self.diversity_matrix[num_trial, num_round, :, :]

    def get_diversity_per_agent(self, agent_id: int, num_trial: int, num_round: int) -> np.array:
        """
        Get the diversity for a specific agent. Sum over the line and the column to get the total diversity of the agent.
        """
        # use the diversity matrix to get the diversity of the agent. Sum over the line and the column where the agent is.

        return (np.sum(self.diversity_matrix[num_trial, num_round, agent_id, :]) + np.sum(self.diversity_matrix[num_trial, num_round, :, agent_id]) ) / (self.num_agents - 1)


    def get_diversity_per_type_of_policy_until_round_specific_trial(self, num_round: int,num_trial : int) -> np.array:
        """
        Get the diversity of each policy type. Diversity are averaged over policies of the same type (std is also computed).
        """
        diversity_per_type = {}
        # get number of agents per type
        agents_per_type = {}
        for agent in self.agents:
            if agent.policy_type not in agents_per_type:
                agents_per_type[agent.policy_type] = 0
            agents_per_type[agent.policy_type] += 1
        diversity_logs_per_type = {policy : 0 for policy in agents_per_type.keys()}
        for agent in self.agents:
            if agent.policy_type not in diversity_per_type:
                diversity_per_type[agent.policy_type] = np.zeros((num_round, agents_per_type[agent.policy_type]))
            # get tbe diversity of the agent until the round num_round
            for i in range(num_round):
                diversity = self.get_diversity_per_agent(agent.id, num_trial, i)
                diversity_per_type[agent.policy_type][i, diversity_logs_per_type[agent.policy_type]] = diversity
            diversity_logs_per_type[agent.policy_type] += 1

        for policy_type, diversities in diversity_per_type.items():
            diversity_per_type[policy_type] = np.array(diversities) / agents_per_type[policy_type]
            mean = np.mean(diversity_per_type[policy_type], axis=1)
            std = np.std(diversity_per_type[policy_type], axis=1)
            diversity_per_type[policy_type] = (mean, std)


        return diversity_per_type


    def get_diversity_per_type_of_policy_all_trial(self,num_round: int, num_trials: int) -> np.array:
        """
        Get the diversity of each policy type. Diversity are averaged over policies of the same type (std is also computed). All is averaged over all trials.
        """
        diversity_per_type = {}
        # get number of agents per type
        agents_per_type = {}
        for agent in self.agents:
            if agent.policy_type not in agents_per_type:
                agents_per_type[agent.policy_type] = 0
            agents_per_type[agent.policy_type] += 1
        diversity_logs_per_type = {policy : 0 for policy in agents_per_type.keys()}
        for agent in self.agents:
            if agent.policy_type not in diversity_per_type:
                diversity_per_type[agent.policy_type] = np.zeros((num_trials,num_round, agents_per_type[agent.policy_type]))
            # get tbe diversity of the agent until the round num_round
            for i in range(num_round):
                for j in range(num_trials):
                    diversity = self.get_diversity_per_agent(agent.id, j, i)
                    diversity_per_type[agent.policy_type][j, i, diversity_logs_per_type[agent.policy_type]] = diversity

            diversity_logs_per_type[agent.policy_type] += 1

        for policy_type, diversities in diversity_per_type.items():
            diversity_per_type[policy_type] = np.array(diversities) / agents_per_type[policy_type]
            # mean over agents
            diversity_per_type[policy_type] = np.mean(diversity_per_type[policy_type], axis=-1)
            # mean over trials
            mean = np.mean(diversity_per_type[policy_type], axis=0)
            std = np.std(diversity_per_type[policy_type], axis=0)

            diversity_per_type[policy_type] = (mean, std)

        return diversity_per_type



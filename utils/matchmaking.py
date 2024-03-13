import numpy as np
from typing import List
from rating.rating import RatingSystem
from Agent import Agent

class Prioritized_fictitious_plays():
    def __init__(self, list_of_agents : List[Agent], p : float):
        """
        This function initializes the matchmaking algorithm with the list of agents and the parameter p.
        Parameters:
            - list_of_agents : List[Agent]
                The list of all the agents
            - p : float
                The parameter p used to calculate the probability of winning
        """
        self.probs_matrix = np.zeros((len(list_of_agents), len(list_of_agents)))
        self.p = p

    def get_probability(self, agent : Agent, list_of_agents : List[Agent], rating_system : RatingSystem):
        """
        This function calculates the probability of winning for an agent against all the other agents in the list
        of agents. The probability is calculated using the rating system and the probabilities are stored in a
        matrix.

        Parameters:
            - agent : Agent
                The agent for which the probability of winning is to be calculated
            - list_of_agents : List[Agent]
                The list of all the agents
            - rating_system : RatingSystem
                The rating system used to calculate the probability of winning

        Returns:
            - np.array
                The probability of winning for the agent against all the other agents


        """

        for opponent in list_of_agents:
            if agent != opponent:
                # Calculate the probability of winning
                probs = rating_system.get_probs_of_winning(agent.id, opponent.id)
                # Update the probability of winning
                self.probs_matrix[agent.id, opponent.id] = probs

        # iterate over all the values
        for line in range(self.probs_matrix.shape[0]):
            for col in range(self.probs_matrix.shape[1]):
                if line != col:
                    self.probs_matrix[line, col] = self.weighted_function(self.probs_matrix[line, col]) / np.sum(
                        self.weighted_function(self.probs_matrix[line, :]))


        return self.probs_matrix[agent.id, :]


    def weighted_function(self, prob : float):
        """
        This function calculates the weighted function of the probability of winning.

        Parameters:
            - prob : float
                The probability of winning

        Returns:
            - float
                The weighted function of the probability of winning

        """
        return (1 - prob) ** self.p

    def update_probability(self, list_of_agents : List[Agent], rating_system : RatingSystem):
        """
         This function updates the probability matrix for all the agents in the list of agents.

         Parameters:
             - list_of_agents : List[Agent]
                 The list of all the agents
             - rating_system : RatingSystem
                 The rating system used to calculate the probability of winning
         Returns:
             - np.array
                 The probability of winning for the agent against all the other agents

        """
        for agent in list_of_agents:
            self.get_probability(agent, list_of_agents, rating_system)

    def sample_opponent_among_probs(self, agent : Agent, list_of_agents : List[Agent]):
        """
        This function samples an opponent for the agent from the list of agents based on the probabilities
        calculated by the get_probability function.

        Parameters:
            - agent : Agent
                The agent for which the opponent is to be sampled
            - list_of_agents : List[Agent]
                The list of all the agents

        Returns:
            - Agent
                The sampled opponent for the agent

        """

        return np.random.choice(list_of_agents, p=self.probs_matrix[agent.id, :])

    def get_opponents(self, agent : Agent, list_of_agents : List[Agent],num_opponents : int) -> List[int]:
        """
        This function samples an opponent for the agent from the list of agents based on the probabilities
        calculated by the get_probability function.

        Parameters:
            - agent : Agent
                The agent for which the opponent is to be sampled
            - list_of_agents : List[Agent]
                The list of all the agents
            - num_opponents : int
                The number of opponents to be sampled

        Returns:
            - Agent
                The sampled opponent for the agent

        """
        opponents_list = []
        while len(opponents_list) < num_opponents:
            opponent = np.random.choice(list_of_agents, p=self.probs_matrix[agent.id, :])
            if opponent not in opponents_list:
                opponents_list.append(opponent.id)

        return opponents_list



    def get_all_opponents(self,list_agent : list[Agent], rating : RatingSystem, num_opponent : int):

        """
        This function returns the list of opponents for all the agents in the list of agents. The number of opponents
        is specified by the num_opponent parameter.

        Parameters:
            - list_agent : list[Agent]
                The list of all the agents
            - rating : RatingSystem
                The rating system used to calculate the probability of winning
            - num_opponent : int
                The number of opponents to be sampled

        Returns:
            - list[list[Agent]]
                The list of opponents for all the agents in the list of agents
        """

        self.update_probability(list_agent, rating)
        opponent_dict = { agent.id : self.get_opponents(agent, list_agent, num_opponent) for agent in list_agent}
        return opponent_dict

    def get_all_pairs(self,list_agent : list[Agent], rating : RatingSystem, num_opponent : int):
        """
        This function returns all the pairs of opponents for all the agents in the list of agents (without repetition).
        The number of opponents is specified by the num_opponent parameter.

        Parameters:
            - list_agent : list[Agent]
                The list of all the agents
            - rating : RatingSystem
                The rating system used to calculate the probability of winning
            - num_opponent : int
                The number of opponents to be sampled

        Returns:
            - list[list[Agent]]
                The list of opponents for all the agents in the list of agents
        """

        self.update_probability(list_agent, rating)
        pairs_dict = { agent.id : self.get_opponents(agent, list_agent, num_opponent) for agent in list_agent}
        # Remove the pairs that are already present in the dictionary
        for agent in list_agent:
            for opponent in pairs_dict[agent.id]:
                if agent.id in pairs_dict[opponent]:
                    pairs_dict[agent.id].remove(opponent)

        assert len(pairs_dict.keys()) == len(list_agent)  , "The number of agents is not correct"
        assert [len(pairs_dict[agent.id]) == num_opponent for agent in list_agent] , "The number of opponents is not correct"
        return pairs_dict

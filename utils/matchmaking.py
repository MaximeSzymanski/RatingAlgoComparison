import numpy as np
from typing import List
from rating.rating import RatingSystem
from Agent import Agent

class Prioritized_fictitious_plays():
    def __init__(self, list_of_agents : List[Agent]):
        self.probs_matrix = np.zeros((len(list_of_agents), len(list_of_agents)))


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
        # Normalize the probabilities
        self.probs_matrix[agent.id, :] /= np.sum(self.probs_matrix[agent.id, :])

        return self.probs_matrix[agent.id, :]

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

    def get_opponent(self, agent : Agent, list_of_agents : List[Agent],num_opponents : int):
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
                opponents_list.append(opponent)

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
        opponent_dict = { agent.id : self.get_opponent(agent, list_agent, num_opponent) for agent in list_agent}
        return opponent_dict

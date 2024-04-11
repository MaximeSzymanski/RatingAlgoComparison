import numpy as np
from typing import List
from poprank import Rate
#from rating.rating import RatingSystem
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
        #probability agent x wins agains agent y
        self.winning_probs_matrix = np.zeros((len(list_of_agents), len(list_of_agents)))

        #probability agent x plays agains agent y
        self.matchmaking_probs_matrix = np.zeros((len(list_of_agents), len(list_of_agents)))
        self.p = p

    def get_probability(self, agent : Agent):
        return self.matchmaking_probs_matrix[agent.id, :]

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
        return (1 - prob) * prob * self.p + 0.01 # for numerical stability

    def update_probability(self, list_of_agents : List[Agent], ratings : "List[Rate]", rating_system : str):
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
            for opponent in list_of_agents:
                # Calculate the probability of winning
                probs = ratings[rating_system][agent.id].predict(ratings[rating_system][opponent.id]) 
                # Update the probability of winning
                self.winning_probs_matrix[agent.id, opponent.id] = probs
        
        # iterate over all the values
        for line in range(self.matchmaking_probs_matrix.shape[0]):
            self.matchmaking_probs_matrix[line] = self.weighted_function(self.winning_probs_matrix[line]) / (np.sum(
                self.weighted_function(self.winning_probs_matrix[line])) - self.weighted_function(self.winning_probs_matrix[line, line]))
            self.matchmaking_probs_matrix[line, line] = 0

    def sample_opponent(self, agent : Agent, list_of_agents : List[Agent]):
        return np.random.choice(list_of_agents, p=self.matchmaking_probs_matrix[agent.id, :]/sum(self.matchmaking_probs_matrix[agent.id, :]))

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
            opponents_list.append(self.sample_opponent(agent, list_of_agents).id)
            if opponents_list[-1] == agent.id:
                raise ValueError("Something is fucked here")

        return opponents_list

    def get_all_opponents(self,list_agent : list[Agent], ratings : "List[Rate]", rating_system : str, num_opponent : int):
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

        self.update_probability(list_agent, ratings, rating_system)
        pairs_dict = { agent.id : self.get_opponents(agent, list_agent, num_opponent) for agent in list_agent}

        assert len(pairs_dict.keys()) == len(list_agent)  , "The number of agents is not correct"
        assert [len(pairs_dict[agent.id]) == num_opponent for agent in list_agent] , "The number of opponents is not correct"
        return pairs_dict

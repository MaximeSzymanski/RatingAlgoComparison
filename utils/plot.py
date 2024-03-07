from typing import List
import matplotlib.pyplot as plt
from Agent import Agent
import numpy as np
from utils.policy import Policy
from typing import List, Dict
def plot_winrate_over_time(agent_1: Agent,agent_2 : Agent, agent_1_win: List[int], agent_2_win: List[int],
                           draws: List[int]) -> None:
    """
    Plots the win rate over time.

    Parameters:
        random_agent (Agent): The random agent used in the simulation.
        random_agent_win (List[int]): List of wins for the random agent.
        opponent_win (List[int]): List of wins for the opponent agent.
        draws (List[int]): List of draws.
    """
    plt.plot(agent_1_win, label=f"{agent_1.policy_type} Win Rate")
    plt.plot(agent_2_win, label=f"{agent_2.policy_type} Win Rate")
    plt.plot(draws, label="Draw Rate")
    plt.xlabel("Number of Fights")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.title("Win Rate Over Time")
    plt.show()


def plot_elo_per_policy(policies: List[str], elos_mean: Dict[str, List[int]], elos_std: Dict[str, List[int]]) -> None:
    """
    Plot the Elo rating of each policy over time.

    Parameters:
    - policies (List[str]): A list of policy names.
    - elos_mean (Dict[str, List[int]]): A dictionary containing the mean Elo ratings for each policy.
    - elos_std (Dict[str, List[int]]): A dictionary containing the standard deviations of Elo ratings for each policy.
    """
    for policy in policies:
        plt.plot(range(len(elos_mean[policy])), elos_mean[policy], label=policy)

        x = np.array(range(len(elos_mean[policy])))
        y_mean = np.array(elos_mean[policy])
        y_std = np.array(elos_std[policy])
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3)

    plt.xlabel("Number of Fights")
    plt.ylabel("Elo Rating")
    plt.legend()
    plt.title("Elo Rating Over Time")
    plt.savefig('elo_rating.png')
    plt.clf()

def plot_strategy_landscape(agent_data : dict,agents_type : List[Policy],index_file=0):
    """
    This function plots the strategy landscape of the agents. There is one data point per agent,
    representing the mean of all actions played by that agent.
    :argument agents_actions: A list of dictionaries, where each dictionary contains the actions played by an agent.
    """
    agent_type_plotted = dict()
    for agent_type in agents_type:
        agent_type_plotted[agent_type] = False
    for (index, agent) , agent_type in zip(enumerate(agent_data), agents_type):
        total_actions = sum(agent.values())
        x_pos = sum((agent[action] / total_actions) * index_action for index_action, action in enumerate(agent.keys()))
        y_pos = max(agent[action] / total_actions for action in agent.keys())
        color = "blue" if agent_type == Policy.PPO else "red"
        if agent_type_plotted[agent_type] == False:
            plt.scatter(x_pos, y_pos, color=color, label=agent_type)
            agent_type_plotted[agent_type] = True
        else:
            plt.scatter(x_pos, y_pos, color=color)

    plt.xlabel("Mean Action Utilization")
    plt.ylabel("Highest Action Percentage")
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(0, len(agent_data[0]), 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title("Strategy Landscape")


    # save the plot in a folder
    plt.savefig("plots/strategy_landscape_" + str(index_file) + ".png")
    # clear the plot
    plt.clf()

def plot_strategy_landscape_elo_fading(agent_data : dict,agents_type : List[Policy],agent_elo : dict = None,index_file=0,):
    """
    This function plots the strategy landscape of the agents. There is one data point per agent,
    representing the mean of all actions played by that agent.
    It also use a fading color to represent the elo of the agent (the higher the elo, the darker the color).
    :argument agents_actions: A list of dictionaries, where each dictionary contains the actions played by an agent.
    :argument agent_elo: A dictionary containing the elo of each agent.
    """
    agent_type_plotted = dict()

    for agent_type in agents_type:
        agent_type_plotted[agent_type] = False
    for (index, agent) , agent_type in zip(enumerate(agent_data), agents_type):
        total_actions = sum(agent.values())
        x_pos = sum((agent[action] / total_actions) * index_action for index_action, action in enumerate(agent.keys()))
        y_pos = max(agent[action] / total_actions for action in agent.keys())
        color = "blue" if agent_type == Policy.PPO else "red"


        if agent_type_plotted[agent_type] == False:
            plt.scatter(x_pos, y_pos, color=color, label=agent_type)
            # round the elo to the nearest integer
            plt.text(x_pos, y_pos, str(round(agent_elo[index])), fontsize=9)
            agent_type_plotted[agent_type] = True
        else:
            plt.scatter(x_pos, y_pos, color=color)
            plt.text(x_pos, y_pos, str(round(agent_elo[index])), fontsize=9)

    plt.xlabel("Mean Action Utilization")
    plt.ylabel("Highest Action Percentage")
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(0, len(agent_data[0]), 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title("Strategy Landscape with ELO fading")
    plt.savefig("strategy_landscape_elo_fading.png")
    plt.clf()
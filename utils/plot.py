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
    plt.xlabel("Number of Rounds")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.title("Win Rate Over Time")
    plt.show()





def plot_strategy_landscape(agent_data : dict,agents_type : List[str],index_file=0):
    """
    This function plots the strategy landscape of the agents. There is one data point per agent,
    representing the mean of all actions played by that agent.
    :argument agents_actions: A list of dictionaries, where each dictionary contains the actions played by an agent.
    """
    agent_type_plotted = dict()
    for agent_type in agents_type:
        agent_type_plotted[agent_type] = False
    colors = plt.cm.get_cmap('tab10', len(agents_type))

    for (index, agent) , agent_type in zip(enumerate(agent_data), agents_type):
        total_actions = sum(agent.values())
        x_pos = sum((agent[action] / total_actions) * index_action for index_action, action in enumerate(agent.keys()))
        y_pos = max(agent[action] / total_actions for action in agent.keys())
        color = colors(agents_type.index(agent_type))
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


def plot_strategy_landscape_elo_fading(agent_data: dict, agents_type: List[str], agent_elo: dict = None, index_file=0):
    """
    This function plots the strategy landscape of the agents. There is one data point per agent,
    representing the mean of all actions played by that agent.
    It also uses fading color to represent the Elo of the agent (the higher the Elo, the darker the color).

    :param agent_data: A dictionary where keys are agent indices and values are dictionaries representing actions played by each agent.
    :param agents_type: A list containing the types of agents.
    :param agent_elo: A dictionary containing the Elo of each agent.
    :param index_file: Index for the saved file (optional).
    """
    list_types = list(set(agents_type))
    agent_type_plotted = {agent_type: False for agent_type in list_types}
    colors = plt.cm.get_cmap('tab20', len(list_types))
    for (index, agent), agent_type in zip(enumerate(agent_data), agents_type):
        total_actions = sum(agent.values())
        x_pos = sum((agent[action] / total_actions) * index_action for index_action, action in enumerate(agent.keys()))
        y_pos = max(agent[action] / total_actions for action in agent.keys())
        color = colors(list_types.index(agent_type))
        if not agent_type_plotted[agent_type]:
            agent_type_plotted[agent_type] = True
            plt.scatter(x_pos, y_pos, color=color, label=agent_type)
        else:
            plt.scatter(x_pos, y_pos, color=color)

        #if agent_elo:
            # Round the Elo to the nearest integer and display it as text
            #lt.text(x_pos, y_pos, str(round(agent_elo[index])), fontsize=9)

    plt.xlabel("Mean Action Utilization")
    plt.ylabel("Highest Action Percentage")
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(0, len(agent_data[0]), 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title("Strategy Landscape with ELO fading")
    plt.savefig(f"strategy_landscape_elo_fading_{index_file}.png")
    plt.clf()


def plot_diversity_matrix(diversity_matrix: np.ndarray, agents_type: List[str], index_file=0) -> None:
    """
    Plot the diversity matrix of the agents.

    Parameters:
        diversity_matrix (np.ndarray): The diversity matrix of the agents.
        agents_type (List[str]): The types of agents.
        index_file (int): The index of the file (optional).
    """
    diversity_matrix = np.triu(diversity_matrix, k=1) + np.triu(diversity_matrix, k=1).T
    plt.figure(figsize=(10, 10))
    plt.imshow(diversity_matrix, cmap='bwr', interpolation='nearest')
    plt.xticks(range(len(agents_type)), agents_type, rotation=90)
    plt.yticks(range(len(agents_type)), agents_type)
    plt.title("Diversity Matrix")
    plt.colorbar()
    plt.show()


def plot_and_save_diversity_matrix(diversity_matrix: np.ndarray, agents: List[Agent], index_file: int) -> None:
        plt.imshow(diversity_matrix, cmap='bwr', interpolation='nearest')
        plt.colorbar()
        plt.xticks(np.arange(len(agents)), [agent.policy_name for agent in agents], rotation=90)
        plt.yticks(np.arange(len(agents)), [agent.policy_name for agent in agents])
        plt.title("Diversity Matrix " + str(index_file))
        plt.savefig("diversity_matrix/diversity_matrix_" + str(index_file) + ".png")
        plt.clf()

def plot_diversity_over_time(diversity_over_time: np.array,number_round=0) -> None:
    """
    Plot the diversity over time.

    Parameters:
        diversity_over_time (np.array): The diversity over time. shape (number_rounds, number_agents)
        number_round (int): The number of rounds to plot (optional).
    """
    if number_round == 1:
        return
    diversity_over_time = diversity_over_time[:number_round]
    print(f"shape of diversity over time: {diversity_over_time.shape}")
    diversity_over_time_mean = np.mean(diversity_over_time, axis=1)
    diversity_over_time_std = np.std(diversity_over_time, axis=1)
    plt.plot(diversity_over_time_mean, label="Mean Diversity")
    plt.fill_between(np.arange(len(diversity_over_time_mean)), diversity_over_time_mean - diversity_over_time_std,
                     diversity_over_time_mean + diversity_over_time_std, alpha=0.3)
    plt.xlabel("Number of Rounds")
    plt.ylabel("Diversity")
    plt.legend()
    plt.title("Diversity Over Time")
    plt.plot()
    plt.show()

def plot_and_save_diversity_over_time(diversity_over_time: np.array, number_round=0) -> None:
    """
    Plot the diversity over time.

    Parameters:
        diversity_over_time (np.array): The diversity over time. shape (number_rounds, number_agents)
        number_round (int): The number of rounds to plot (optional).
    """
    if number_round <= 1:
        return
    diversity_over_time = diversity_over_time[:number_round]
    print(f"shape of diversity over time: {diversity_over_time.shape}")
    diversity_over_time_mean = np.mean(diversity_over_time, axis=1)
    diversity_over_time_std = np.std(diversity_over_time, axis=1)
    plt.plot(diversity_over_time_mean, label="Mean Diversity")
    plt.fill_between(np.arange(len(diversity_over_time_mean)), diversity_over_time_mean - diversity_over_time_std,
                     diversity_over_time_mean + diversity_over_time_std, alpha=0.3)
    plt.xlabel("Number of Rounds")
    plt.ylabel("Diversity")
    plt.legend()
    plt.title("Diversity Over Time")
    plt.plot()
    plt.savefig("diversity_score/diversity_score_" + str(number_round) + ".png")
    plt.clf()

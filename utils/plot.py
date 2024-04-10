import os
import matplotlib.pyplot as plt
import numpy as np
from Agent import Agent
from utils.policy import Policy
from typing import List, Dict


def plot_winrate_over_time(agent_1: Agent, agent_2: Agent, agent_1_win: List[int], agent_2_win: List[int],
                           draws: List[int]) -> None:
    """
    Plots the win rate over time.

    Parameters:
        agent_1 (Agent): The first agent.
        agent_2 (Agent): The second agent.
        agent_1_win (List[int]): List of wins for the first agent.
        agent_2_win (List[int]): List of wins for the second agent.
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


def plot_strategy_landscape(agent_data: Dict[int, Dict[int, int]], agents_type: List[str], index_file: int = 0) -> None:
    """
    Plot the strategy landscape of the agents.

    Parameters:
        agent_data (Dict[int, Dict[int, int]]): A dictionary where keys are agent indices and values are dictionaries
            representing actions played by each agent.
        agents_type (List[str]): A list containing the types of agents.
        index_file (int): Index for the saved file (optional).
    """
    agent_type_plotted = {agent_type: False for agent_type in agents_type}
    colors = plt.cm.get_cmap('tab10', len(agents_type))

    for (index, agent), agent_type in zip(enumerate(agent_data), agents_type):
        total_actions = sum(agent.values())
        x_pos = sum((agent[action] / total_actions) *
                    index_action for index_action, action in enumerate(agent.keys()))
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
    plt.savefig(f"plots/strategy_landscape_{index_file}.png")
    plt.clf()


def plot_strategy_landscape_elo_fading(agent_data: Dict[int, Dict[int, int]], agents_type: List[str],
                                       agent_elo: Dict[int, int] = None, index_file: int = 0) -> None:
    """
    Plot the strategy landscape of the agents with Elo fading.

    Parameters:
        agent_data (Dict[int, Dict[int, int]]): A dictionary where keys are agent indices and values are dictionaries
            representing actions played by each agent.
        agents_type (List[str]): A list containing the types of agents.
        agent_elo (Dict[int, int]): A dictionary containing the Elo of each agent (optional).
        index_file (int): Index for the saved file (optional).
    """
    list_types = list(set(agents_type))
    agent_type_plotted = {agent_type: False for agent_type in list_types}
    colors = plt.cm.get_cmap('tab20', len(list_types))
    for (index, agent), agent_type in zip(enumerate(agent_data), agents_type):
        total_actions = sum(agent.values())
        x_pos = sum((agent[action] / total_actions) *
                    index_action for index_action, action in enumerate(agent.keys()))
        y_pos = max(agent[action] / total_actions for action in agent.keys())
        color = colors(list_types.index(agent_type))
        if not agent_type_plotted[agent_type]:
            agent_type_plotted[agent_type] = True
            plt.scatter(x_pos, y_pos, color=color, label=agent_type)
        else:
            plt.scatter(x_pos, y_pos, color=color)

    plt.xlabel("Mean Action Utilization")
    plt.ylabel("Highest Action Percentage")
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(0, len(agent_data[0]), 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title("Strategy Landscape with ELO fading")
    plt.savefig(f"plots/strategy_landscape_elo_fading_{index_file}.png")
    plt.clf()


def plot_diversity_matrix(diversity_matrix: np.array, index_file: int,agents : list[Agent] ,path : str) -> None:
    """
    Plot the diversity matrix.

    Parameters:
        diversity_matrix (np.array): The diversity matrix.
        index_file (int): Index for the saved file (optional).
    """



    plt.imshow(diversity_matrix, cmap='bwr', interpolation='nearest')
    # plot the id and the policy type of the agents in the x and y axis
    plt.xticks(range(len(agents)), [f"{agent.id} - {agent.policy_type}" for agent in agents], rotation=90)
    plt.yticks(range(len(agents)), [f"{agent.id} - {agent.policy_type}" for agent in agents])

    plt.colorbar()
    plt.title("Diversity Matrix")
    plt.savefig(f"{path}/diversity_matrix_{index_file}.png")
    plt.clf()

def plot_diversity_per_policy_round(diversity_per_policy: Dict[Policy, tuple[np.array,np.array]], index_file: int, path : str) -> None:
    """
    Plot the diversity per policy.

    Parameters:
        diversity_per_policy (Dict[Policy, np.array]): A dictionary where keys are policies and values are (mean, std) tuples.
        index_file (int): Index for the saved file (optional).
    """

    for policy, (mean, std) in diversity_per_policy.items():
        # plot std as a shadow
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.4)
        plt.plot(range(len(mean)), mean, label=f"{policy}")

    plt.xlabel("Number of Rounds")
    plt.ylabel("Diversity")
    plt.legend()
    plt.title("Diversity Per Policy until Round {}".format(index_file))
    plt.savefig(f"{path}/diversity_per_policy_{index_file}.png")
    plt.clf()

def plot_diversity_per_type_of_policy_averaged_over_trials(diversity_per_type: Dict[str, tuple[np.array,np.array]], path : str) -> None:
    """
    Plot the diversity per type of policy.

    Parameters:
        diversity_per_type (Dict[str, np.array]): A dictionary where keys are policy types and values are (mean, std) tuples.
        index_file (int): Index for the saved file (optional).
    """
    for policy_type, (mean, std) in diversity_per_type.items():
        # plot std as a shadow
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.4)
        plt.plot(range(len(mean)), mean, label=f"{policy_type}")
    plt.xlabel("Number of Rounds")
    plt.ylabel("Diversity")
    plt.legend()
    plt.title("Diversity Per Type of Policy Averaged Over Trials")
    plt.savefig(f"{path}/diversity_per_type_of_policy.png")
    plt.clf()

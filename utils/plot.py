from typing import List
import matplotlib.pyplot as plt
from Agent import Agent

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
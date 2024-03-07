from Agent import Agent
import numpy as np
from models.PPO import PPO
from models.DQN import DQN
from pettingzoo.utils.env import AECEnv
from pettingzoo.classic import connect_four_v3, tictactoe_v3,texas_holdem_v4
import matplotlib.pyplot as plt
from typing import List


class Population():
    def __init__(self, env: AECEnv, num_agents) -> None:
        self.agents: list[Agent] = []
        self.env: AECEnv = env
        self.state_size = 72

        self.action_size = env.action_space("player_1").n

    def add_agent(self, policy_type):
        self.agents.append(Agent(policy_type, self.state_size, self.action_size))

    def train_fight_1vs1(self, num_fights=1000, agent_1_index=0, agent_2_index=1):
        """
        Simulates fights between a player agent and a randomly selected opponent agent.

        Parameters:
            num_fights (int): Number of fights to simulate. Defaults to 2000.
        """
        # get a random agent
        agent_1: Agent = self.agents[agent_1_index]
        agent_2: Agent = self.agents[agent_2_index]

        agent_1_win = []
        agent_2_win = []
        draws = []
        past_state_agent_1 = None
        past_action_agent_1 = None
        past_reward_agent_1 = None
        past_mask_agent_1 = None
        past_value_agent_1 = None
        past_log_prob_agent_1 = None
        past_done_agent_1 = None

        past_state_agent_2 = None
        past_action_agent_2 = None
        past_reward_agent_2 = None
        past_mask_agent_2 = None
        past_value_agent_2 = None
        past_log_prob_agent_2 = None
        past_done_agent_2 = None

        for fight in range(num_fights):
            # our agent is player 1 and the random bot is player 2

            self.env.reset(seed=42)
            current_episode_reward_agent_1 = 0
            current_episode_reward_agent_2 = 0
            draw_count = 0
            step = 0
            update_freq = 4
            for agent in self.env.agent_iter():
                # check if the agent can train
                if agent == "player_0" and agent_1.policy.experience_replay.can_train() and step % update_freq == 0:
                    agent_1.policy.train_agent()
                    pass
                elif agent == "player_1" and agent_2.policy.experience_replay.can_train() and step % update_freq == 0:
                    # agent_2.policy.train_agent()
                    pass

                observation, reward, termination, truncation, info = self.env.last()

                # check if the agent is the random agent
                if agent == "player_0":
                    if past_state_agent_1 is not None and past_action_agent_1 is not None and past_reward_agent_1 is not None and past_mask_agent_1 is not None:
                        next_state = observation["observation"]
                        # flatten
                        next_state = next_state.flatten()
                        if agent_1.policy_type == "PPO":
                            agent_1.policy.experience_replay.add_step(state=past_state_agent_1,
                                                                      action=past_action_agent_1,
                                                                      reward=past_reward_agent_1,
                                                                      next_state=next_state,
                                                                      done=past_done_agent_1,
                                                                      old_log_prob=past_log_prob_agent_1,
                                                                      value=past_value_agent_1,
                                                                      action_mask=past_mask_agent_1)
                        elif agent_1.policy_type == "DQN":

                            agent_1.policy.experience_replay.add_step(state=past_state_agent_1,
                                                                      action=past_action_agent_1,
                                                                      reward=past_reward_agent_1,
                                                                      next_state=next_state, done=past_done_agent_1,
                                                                      action_mask=past_mask_agent_1)

                    past_state_agent_1 = observation["observation"].flatten()
                    past_reward_agent_1 = reward
                    current_episode_reward_agent_1 += reward
                elif agent == "player_1":
                    if past_state_agent_2 is not None and past_action_agent_2 is not None and past_reward_agent_2 is not None and past_mask_agent_2 is not None:
                        next_state = observation["observation"]
                        # flatten
                        next_state = next_state.flatten()
                        if agent_2.policy_type == "PPO":
                            """agent_2.policy.experience_replay.add_step(state=past_state_agent_2,
                                                                        action=past_action_agent_2,
                                                                        reward=past_reward_agent_2,
                                                                        next_state=next_state,
                                                                        done=past_done_agent_2,
                                                                        old_log_prob=past_log_prob_agent_2,
                                                                        value=past_value_agent_2,
                                                                        action_mask=past_mask_agent_2)"""
                            pass
                        elif agent_2.policy_type == "DQN":

                            agent_2.policy.experience_replay.add_step(state=past_state_agent_2,
                                                                      action=past_action_agent_2,
                                                                      reward=past_reward_agent_2,
                                                                      next_state=next_state, done=past_done_agent_2,
                                                                      action_mask=past_mask_agent_2)

                    past_state_agent_2 = observation["observation"].flatten()
                    past_reward_agent_2 = reward
                    current_episode_reward_agent_2 += reward

                if termination or truncation:
                    action = None

                else:
                    mask = observation["action_mask"]
                    # this is where you would insert your policy
                    if agent == "player_0":
                        state = observation["observation"]
                        state = state.flatten()
                        if agent_1.policy_type == "PPO":
                            action, log_prob, value = agent_1.policy.get_action(state, mask)
                            past_log_prob_agent_1 = log_prob
                            past_value_agent_1 = value
                        elif agent_1.policy_type == "DQN":
                            action = agent_1.policy.act(state=state, mask=mask)
                        past_action_agent_1 = action
                        past_mask_agent_1 = mask
                        past_state_agent_1 = state

                        past_done_agent_1 = termination or truncation

                    elif agent == "player_1":
                        state = observation["observation"]
                        state = state.flatten()
                        if agent_2.policy_type == "PPO":
                            action, log_prob, value = agent_2.policy.get_action(state, mask)
                            past_log_prob_agent_2 = log_prob
                            past_value_agent_2 = value
                        elif agent_2.policy_type == "DQN":
                            action = agent_2.policy.act(state=state, mask=mask)
                        past_action_agent_2 = action
                        past_mask_agent_2 = mask
                        past_state_agent_2 = state

                        past_done_agent_2 = termination or truncation

                self.env.step(action)
            if fight % 100 == 0:
                if agent_2.policy_type == "DQN":
                    agent_2.policy.update_epsilon()
                elif agent_1.policy_type == "DQN":
                    agent_1.policy.update_epsilon()
            self.compute_winner(agent_1_win, agent_2_win, draws, current_episode_reward_agent_1,
                                current_episode_reward_agent_2)
            agent_1.policy.writer.add_scalar("Reward", current_episode_reward_agent_1, fight)
            agent_2.policy.writer.add_scalar("Reward", current_episode_reward_agent_2, fight)
            self.env.close()

        # Calculating win rates over time
        agent_1_win, agent_2_win, draws = self.compute_winrate_over_time(num_fights, agent_1_win, agent_2_win, draws)

        # Plotting win rates over time
        self.plot_winrate_over_time_1v1(agent_1, agent_2, agent_1_win, agent_2_win, draws)
        self.test_fight_1vs1(2000, agent_1_index=agent_1_index, agent_2_index=agent_2_index)

    def test_fight_1vs1(self, num_fights=2000, agent_1_index=0, agent_2_index=1):
        """
        Simulates fights between a player agent and a randomly selected opponent agent.

        Parameters:
            num_fights (int): Number of fights to simulate. Defaults to 2000.
        """
        # get a random agent
        agent_1: Agent = self.agents[agent_1_index]
        agent_2: Agent = self.agents[agent_2_index]

        agent_1_win = []
        agent_2_win = []
        draws = []
        past_state_agent_1 = None
        past_action_agent_1 = None
        past_reward_agent_1 = None
        past_mask_agent_1 = None
        past_value_agent_1 = None
        past_log_prob_agent_1 = None
        past_done_agent_1 = None

        past_state_agent_2 = None
        past_action_agent_2 = None
        past_reward_agent_2 = None
        past_mask_agent_2 = None
        past_value_agent_2 = None
        past_log_prob_agent_2 = None
        past_done_agent_2 = None

        for fight in range(num_fights):
            # our agent is player 1 and the random bot is player 2

            self.env.reset(seed=42)
            current_episode_reward_agent_1 = 0
            current_episode_reward_agent_2 = 0
            draw_count = 0
            step = 0
            update_freq = 4
            for agent in self.env.agent_iter():
                # check if the agent can train

                observation, reward, termination, truncation, info = self.env.last()

                # check if the agent is the random agent
                if agent == "player_0":
                    if past_state_agent_1 is not None and past_action_agent_1 is not None and past_reward_agent_1 is not None and past_mask_agent_1 is not None:
                        next_state = observation["observation"]
                        # flatten
                        next_state = next_state.flatten()
                        if agent_1.policy_type == "PPO":
                            agent_1.policy.experience_replay.add_step(state=past_state_agent_1,
                                                                      action=past_action_agent_1,
                                                                      reward=past_reward_agent_1,
                                                                      next_state=next_state,
                                                                      done=past_done_agent_1,
                                                                      old_log_prob=past_log_prob_agent_1,
                                                                      value=past_value_agent_1,
                                                                      action_mask=past_mask_agent_1)
                        elif agent_1.policy_type == "DQN":

                            agent_1.policy.experience_replay.add_step(state=past_state_agent_1,
                                                                      action=past_action_agent_1,
                                                                      reward=past_reward_agent_1,
                                                                      next_state=next_state, done=past_done_agent_1,
                                                                      action_mask=past_mask_agent_1)

                    past_state_agent_1 = observation["observation"].flatten()
                    past_reward_agent_1 = reward
                elif agent == "player_1":
                    if past_state_agent_2 is not None and past_action_agent_2 is not None and past_reward_agent_2 is not None and past_mask_agent_2 is not None:
                        next_state = observation["observation"]
                        # flatten
                        next_state = next_state.flatten()

                    past_state_agent_2 = observation["observation"].flatten()
                    past_reward_agent_2 = reward


                if termination or truncation:
                    action = None
                    if agent == "player_0":
                        current_episode_reward_agent_1 += reward
                    else:

                        current_episode_reward_agent_2 += reward
                else:
                    mask = observation["action_mask"]
                    # this is where you would insert your policy
                    if agent == "player_0":
                        state = observation["observation"]
                        state = state.flatten()
                        if agent_1.policy_type == "PPO":
                            action, log_prob, value = agent_1.policy.get_action(state, mask, deterministic=True)
                            past_log_prob_agent_1 = log_prob
                            past_value_agent_1 = value
                        elif agent_1.policy_type == "DQN":
                            action = agent_1.policy.act(state=state, mask=mask, deterministic=True)
                        past_action_agent_1 = action
                        past_mask_agent_1 = mask
                        past_state_agent_1 = state

                        past_done_agent_1 = termination or truncation

                    elif agent == "player_1":
                        state = observation["observation"]
                        state = state.flatten()
                        if agent_2.policy_type == "PPO":
                            action, log_prob, value = agent_2.policy.get_action(state, mask, deterministic=True)
                            past_log_prob_agent_2 = log_prob
                            past_value_agent_2 = value
                        elif agent_2.policy_type == "DQN":
                            action = agent_2.policy.act(state=state, mask=mask, deterministic=True)
                        past_action_agent_2 = action
                        past_mask_agent_2 = mask
                        past_state_agent_2 = state

                        past_done_agent_2 = termination or truncation

                self.env.step(action)

            self.compute_winner(agent_1_win, agent_2_win, draws, current_episode_reward_agent_1,
                                current_episode_reward_agent_2)

            self.env.close()

        # Calculating win rates over time
        agent_1_win, agent_2_win, draws = self.compute_winrate_over_time(num_fights, agent_1_win, agent_2_win, draws)

        # Plotting win rates over time
        self.plot_winrate_over_time_1v1(agent_1, agent_2, agent_1_win, agent_2_win, draws)

    def fight_agent_against_random(self, num_fights: int = 2000) -> None:
        """
        Simulates fights between a player agent and a randomly selected opponent agent.

        Parameters:
            num_fights (int): Number of fights to simulate. Defaults to 2000.
        """
        # get a random agent
        random_agent: Agent = self.agents[np.random.randint(len(self.agents))]
        print("Random agent rating: ", random_agent.rating)
        random_agent_win = []
        opponent_win = []
        draws = []
        past_state = None
        past_action = None
        past_reward = None
        past_mask = None
        past_value = None
        past_log_prob = None
        past_done = None

        for fight in range(num_fights):
            # our agent is player 1 and the random bot is player 2

            self.env.reset(seed=42)
            current_episode_reward = 0
            current_episode_reward_opponent = 0
            draw_count = 0
            step = 0
            update_freq = 4
            for agent in self.env.agent_iter():
                # check if the agent can train
                if agent == "player_1" and random_agent.policy.experience_replay.can_train() and step % update_freq == 0:
                    random_agent.policy.train_agent()

                observation, reward, termination, truncation, info = self.env.last()

                # check if the agent is the random agent
                if agent == "player_1":
                    if past_state is not None and past_action is not None and past_reward is not None and past_mask is not None:
                        next_state = observation["observation"]
                        # flatten
                        next_state = next_state.flatten()
                        if random_agent.policy_type == "PPO":
                            random_agent.policy.experience_replay.add_step(state=past_state,
                                                                           action=past_action,
                                                                           reward=past_reward,
                                                                           next_state=next_state,
                                                                           done=past_done,
                                                                           old_log_prob=past_log_prob,
                                                                           value=past_value,
                                                                           action_mask=past_mask)
                        elif random_agent.policy_type == "DQN":

                            random_agent.policy.experience_replay.add_step(state=past_state, action=past_action,
                                                                           reward=past_reward,
                                                                           next_state=next_state, done=past_done,
                                                                           action_mask=past_mask)

                    past_state = observation["observation"].flatten()
                    past_reward = reward
                    current_episode_reward += reward
                else:
                    current_episode_reward_opponent += reward

                if termination or truncation:
                    action = None

                else:
                    mask = observation["action_mask"]
                    # this is where you would insert your policy
                    if agent == "player_1":
                        state = observation["observation"]
                        state = state.flatten()
                        if random_agent.policy_type == "PPO":
                            action, log_prob, value = random_agent.policy.get_action(state, mask)
                            past_log_prob = log_prob
                            past_value = value
                        elif random_agent.policy_type == "DQN":
                            action = random_agent.policy.act(state=state, mask=mask)
                        past_action = action
                        past_mask = mask
                        past_state = state

                        past_done = termination or truncation

                    else:
                        action = self.env.action_space(agent).sample(mask)

                self.env.step(action)
            if fight % 20 == 0 and random_agent.policy_type == "DQN":
                random_agent.policy.update_epsilon()
            self.compute_winner(random_agent_win, opponent_win, draws, current_episode_reward,
                                current_episode_reward_opponent)
            random_agent.policy.writer.add_scalar("Reward", current_episode_reward, fight)
            self.env.close()

        # Calculating win rates over time
        random_agent_win, opponent_win, draws = self.compute_winrate_over_time(num_fights, random_agent_win,
                                                                               opponent_win, draws)

        # Plotting win rates over time
        #self.plot_winrate_over_time(random_agent, random_agent_win, opponent_win, draws)

    def plot_winrate_over_time(self, random_agent: Agent, random_agent_win: List[int], opponent_win: List[int],
                               draws: List[int]) -> None:
        """
        Plots the win rate over time.

        Parameters:
            random_agent (Agent): The random agent used in the simulation.
            random_agent_win (List[int]): List of wins for the random agent.
            opponent_win (List[int]): List of wins for the opponent agent.
            draws (List[int]): List of draws.
        """
        plt.plot(random_agent_win, label=f"{random_agent.policy_type} Win Rate")
        plt.plot(opponent_win, label="Opponent Win Rate")
        plt.plot(draws, label="Draw Rate")
        plt.xlabel("Number of Fights")
        plt.ylabel("Win Rate")
        plt.legend()
        plt.title("Win Rate Over Time")
        plt.show()

    def plot_winrate_over_time_1v1(self, agent_1: Agent, agent_2: Agent, agent_1_win: List[int], agent_2_win: List[int],
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

    def compute_winrate_over_time(self, num_fights: int, random_agent_win: List[int], opponent_win: List[int],
                                  draws: List[int]) -> tuple[List[int], List[int], List[int]]:
        """
        Computes win rates over time.

        Parameters:
            num_fights (int): Number of fights.
            random_agent_win (List[int]): List of wins for the random agent.
            opponent_win (List[int]): List of wins for the opponent agent.
            draws (List[int]): List of draws.

        Returns:
            Tuple of lists containing win rates for the random agent, opponent agent, and draws.
        """
        random_agent_win = np.cumsum(random_agent_win) / (np.arange(num_fights) + 1)
        opponent_win = np.cumsum(opponent_win) / (np.arange(num_fights) + 1)
        draws = np.cumsum(draws) / (np.arange(num_fights) + 1)
        return random_agent_win, opponent_win, draws

    def compute_winner(self, random_agent_win: List[int], opponent_win: List[int], draws: List[int],
                       current_episode_reward: int, current_episode_reward_opponent: int) -> None:
        """
        Computes the winner of a fight.

        Parameters:
            random_agent_win (List[int]): List of wins for the random agent.
            opponent_win (List[int]): List of wins for the opponent agent.
            draws (List[int]): List of draws.
            current_episode_reward (int): Reward obtained by the player agent.
            current_episode_reward_opponent (int): Reward obtained by the opponent agent.
        """
        if current_episode_reward > current_episode_reward_opponent:
            random_agent_win.append(1)
            opponent_win.append(0)
            draws.append(0)
        elif current_episode_reward < current_episode_reward_opponent:
            random_agent_win.append(0)
            opponent_win.append(1)
            draws.append(0)
        else:
            random_agent_win.append(0)
            opponent_win.append(0)
            draws.append(1)

    def test_agents(self, num_tests: int = 1000) -> None:
        """
        Tests the agents against each other.

        Parameters:
            num_tests (int): Number of tests to conduct. Defaults to 1000.
        """
        # get a random agent
        random_agent: Agent = self.agents[np.random.randint(len(self.agents))]
        print("Random agent rating: ", random_agent.rating)
        random_agent_win = []
        opponent_win = []
        draws = []
        action_list = dict()
        for i in range(random_agent.action_size):
            action_list[f"Action {i}"] = 0

        for fight in range(num_tests):
            # our agent is player 1 and the random bot is player 2
            past_state = None
            past_action = None
            past_reward = None
            past_mask = None
            past_value = None
            past_log_prob = None
            past_done = None
            self.env.reset(seed=42)
            current_episode_reward = 0
            current_episode_reward_opponent = 0
            draw_count = 0

            for agent in self.env.agent_iter():
                # check if the agent can train

                observation, reward, termination, truncation, info = self.env.last()

                # check if the agent is the random agent
                if agent == "player_1":
                    if past_state is not None and past_action is not None and past_reward is not None and past_mask is not None:
                        next_state = observation["observation"]
                        # flatten
                        next_state = next_state.flatten()

                    past_state = observation["observation"]
                    past_reward = reward
                    current_episode_reward += reward
                else:
                    current_episode_reward_opponent += reward

                if termination or truncation:
                    action = None
                else:
                    mask = observation["action_mask"]
                    # this is where you would insert your policy
                    if agent == "player_1":
                        state = observation["observation"]
                        state = state.flatten()
                        if random_agent.policy_type == "PPO":
                            action, log_prob, value = random_agent.policy.get_action(state, mask, deterministic=True)
                            past_log_prob = log_prob
                            past_value = value
                        elif random_agent.policy_type == "DQN":
                            action = random_agent.policy.act(state=state, mask=mask, deterministic=True)
                        past_action = action
                        past_mask = mask
                        past_state = state

                        past_done = termination or truncation
                        current_episode_reward += reward
                        action_list[f"Action {action}"] += 1
                    else:
                        action = self.env.action_space(agent).sample(mask)

                self.env.step(action)

            if current_episode_reward > current_episode_reward_opponent:
                random_agent_win.append(1)
                opponent_win.append(0)
                draws.append(0)
            elif current_episode_reward < current_episode_reward_opponent:
                random_agent_win.append(0)
                opponent_win.append(1)
                draws.append(0)
            else:
                random_agent_win.append(0)
                opponent_win.append(0)
                draws.append(1)

            self.env.close()

        # Calculating win rates over time
        random_agent_win, opponent_win, draws = self.compute_winrate_over_time(num_tests, random_agent_win, opponent_win,
                                                                               draws)
        # Plotting win rates over time
        #self.plot_winrate_over_time(random_agent, random_agent_win, opponent_win, draws)

        return action_list

    def plot_strategy_landscape(self,agent_data : dict,agents_type : List[str],index_file=0):
        """
        This function plots the strategy landscape of the agents. There is one data point per agent,
        representing the mean of all actions played by that agent.
        :argument agents_actions: A list of dictionaries, where each dictionary contains the actions played by an agent.
        """

        for (index, agent) , agent_type in zip(enumerate(agent_data), agents_type):
            total_actions = sum(agent.values())
            x_pos = sum((agent[action] / total_actions) * index_action for index_action, action in enumerate(agent.keys()))
            y_pos = max(agent[action] / total_actions for action in agent.keys())
            if agent_type == "PPO":
                plt.scatter(x_pos, y_pos, label=f"Agent {index}", color="blue")
            elif agent_type == "DQN":
                plt.scatter(x_pos, y_pos, label=f"Agent {index}", color="red")


        plt.xlabel("Mean Action Utilization")
        plt.ylabel("Highest Action Percentage")
        plt.grid()
        plt.xticks(np.arange(0, len(agent_data[0]), 1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.title("Strategy Landscape")
        # save the plot in a folder
        plt.savefig("plots/strategy_landscape_" + str(index_file) + ".png")
        # clear the plot
        plt.clf()



texas_population = Population(texas_holdem_v4.env(), 1)
number_DQN = 10
number_PPO = 10
for i in range(number_DQN):
    texas_population.add_agent("DQN")
for i in range(number_PPO):
    texas_population.add_agent("PPO")
#texas_population.fight_agent_against_random(1000)
list = []
for num_train in range(10):
    print(f"Training {num_train} / 10")
    for index_agent, agent in enumerate(texas_population.agents):
        print(f"Agent {index_agent} / {len(texas_population.agents)}")
        # train
        texas_population.fight_agent_against_random(1000)
        list.append(texas_population.test_agents(1000))
    texas_population.plot_strategy_landscape(list, [agent.policy_type for agent in texas_population.agents], num_train)
#texas_population.add_agent("PPO")

#texas_population.train_fight_1vs1()
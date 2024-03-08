from Agent import Agent
import numpy as np
from models.PPO import PPO
from models.DQN import DQN
from pettingzoo.utils.env import AECEnv
from pettingzoo.classic import connect_four_v3, tictactoe_v3,texas_holdem_v4
import matplotlib.pyplot as plt
from utils.plot import plot_winrate_over_time,plot_elo_per_policy,plot_strategy_landscape,plot_strategy_landscape_elo_fading
from rating.elo import Elo
from utils.policy import Policy
from tqdm import tqdm
from typing import List, Dict

class Population():
    def __init__(self, env: AECEnv, num_agents) -> None:
        self.agents: list[Agent] = []
        self.env: AECEnv = env
        self.state_size = 84
        self.rating = Elo()
        self.base_rating = 1500
        self.action_size = env.action_space("player_1").n

    def add_agent(self, policy_type : str) -> None:
        """
        Add an agent to the population.
        :param policy_type: The type of policy to use for the agent.
        :return: None
        """
        self.agents.append(Agent(policy_type, self.state_size, self.action_size,id=len(self.agents)-1))
        self.rating.add_player(len(self.agents)-1, self.base_rating)

    def remove_agent(self, agent: Agent) -> None:
        """
        Remove an agent from the population.
        :param agent: The agent to remove.
        :return: None
        """
        self.rating.remove_player(agent.id)
        self.agents.remove(agent)



    def training_loop(self, number_round=100) -> None:
        """
        The training loop for the population.
        :param number_round: The number of rounds to train for.
        :return: None
        """



        elo_per_policy_mean = {Policy.PPO: [], Policy.DQN: []}
        elo_per_policy_std = {Policy.PPO: [], Policy.DQN: []}
        for round in tqdm(range(number_round)):
            elo_per_policy_mean_round = {Policy.PPO: [], Policy.DQN: []}
            elo_per_policy_std_round = {Policy.PPO: [], Policy.DQN: []}
            paired_agents = self.rating.find_similar_elo_pairs()
            for agent_1, agent_2 in paired_agents:
                _, _, _ = self.train_fight_1vs1(agent_1_index=agent_1, agent_2_index=agent_2,num_fights=10)
                agent_1_win, agent_2_win, draws = self.test_fight_1vs1(num_fights=1, agent_1_index=agent_1,
                                                                       agent_2_index=agent_2)
                winner = agent_1 if agent_1_win > agent_2_win else agent_2

                loser = agent_1 if agent_1_win < agent_2_win else agent_2

                self.rating.update_ratings(winner, loser)

            for agent in self.agents:
                for policy in [Policy.PPO, Policy.DQN]:
                    if agent.policy_type == policy:
                        elo_per_policy_mean_round[policy].append(self.rating.get_rating(agent.id))
            for policy in [Policy.PPO, Policy.DQN]:
                elo_per_policy_mean[policy].append(np.mean(elo_per_policy_mean_round[policy]))
                elo_per_policy_std[policy].append(np.std(elo_per_policy_mean_round[policy]))




            self.rating.plot_elo_distribution(round)

        plot_elo_per_policy([Policy.PPO, Policy.DQN], elo_per_policy_mean, elo_per_policy_std)

    def train_fight_1vs1(self, num_fights : int =1000, agent_1_index : int =0, agent_2_index : int =1):
        """
        Simulates fights between two agents.
        :param num_fights: Number of fights to simulate. Defaults to 2000.
        :param agent_1_index: Index of the first agent.
        :param agent_2_index: Index of the second agent.
        :return:
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
                    agent_2.policy.train_agent()
                    pass

                observation, reward, termination, truncation, info = self.env.last()

                # check if the agent is the random agent
                if agent == "player_0":
                    if past_state_agent_1 is not None and past_action_agent_1 is not None and past_reward_agent_1 is not None and past_mask_agent_1 is not None:
                        next_state = observation["observation"]
                        # flatten
                        next_state = next_state.flatten()
                        if agent_1.policy_type == Policy.PPO:
                            agent_1.policy.experience_replay.add_step(state=past_state_agent_1,
                                                                      action=past_action_agent_1,
                                                                      reward=past_reward_agent_1,
                                                                      next_state=next_state,
                                                                      done=past_done_agent_1,
                                                                      old_log_prob=past_log_prob_agent_1,
                                                                      value=past_value_agent_1,
                                                                      action_mask=past_mask_agent_1)
                        elif agent_1.policy_type == Policy.DQN:

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
                        if agent_2.policy_type == Policy.PPO:
                            agent_2.policy.experience_replay.add_step(state=past_state_agent_2,
                                                                        action=past_action_agent_2,
                                                                        reward=past_reward_agent_2,
                                                                        next_state=next_state,
                                                                        done=past_done_agent_2,
                                                                        old_log_prob=past_log_prob_agent_2,
                                                                        value=past_value_agent_2,
                                                                        action_mask=past_mask_agent_2)
                            pass
                        elif agent_2.policy_type == Policy.DQN:

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
                        if agent_1.policy_type == Policy.PPO:
                            action, log_prob, value = agent_1.policy.get_action(state, mask)
                            past_log_prob_agent_1 = log_prob
                            past_value_agent_1 = value
                        elif agent_1.policy_type == Policy.DQN:
                            action = agent_1.policy.act(state=state, mask=mask)
                        past_action_agent_1 = action
                        past_mask_agent_1 = mask
                        past_state_agent_1 = state

                        past_done_agent_1 = termination or truncation

                    elif agent == "player_1":
                        state = observation["observation"]
                        state = state.flatten()
                        if agent_2.policy_type == Policy.PPO:
                            action, log_prob, value = agent_2.policy.get_action(state, mask)
                            past_log_prob_agent_2 = log_prob
                            past_value_agent_2 = value
                        elif agent_2.policy_type == Policy.DQN:
                            action = agent_2.policy.act(state=state, mask=mask)
                        past_action_agent_2 = action
                        past_mask_agent_2 = mask
                        past_state_agent_2 = state

                        past_done_agent_2 = termination or truncation

                self.env.step(action)
            if fight % 100 == 0:
                if agent_2.policy_type == Policy.DQN:
                    agent_2.policy.update_epsilon()
                if agent_1.policy_type == Policy.DQN:
                    agent_1.policy.update_epsilon()
            self.compute_winner(agent_1_win, agent_2_win, draws, current_episode_reward_agent_1,
                                current_episode_reward_agent_2)
            agent_1.policy.writer.add_scalar("Reward", current_episode_reward_agent_1, fight)
            agent_2.policy.writer.add_scalar("Reward", current_episode_reward_agent_2, fight)
            self.env.close()

        # Calculating win rates over time
        agent_1_win, agent_2_win, draws = self.compute_winrate_over_time(num_fights, agent_1_win, agent_2_win, draws)

        # Plotting win rates over time
        #self.plot_winrate_over_time_1v1(agent_1, agent_2, agent_1_win, agent_2_win, draws)
        #self.test_fight_1vs1(2000, agent_1_index=agent_1_index, agent_2_index=agent_2_index)

        return agent_1_win, agent_2_win, draws
    def test_fight_1vs1(self, num_fights : int = 2000, agent_1_index : int = 0, agent_2_index : int = 1) -> None:
        """
        Simulates fights between two agents, without training in a deterministic manner.
        :param num_fights: Number of fights to simulate. Defaults to 2000.
        :param agent_1_index: Index of the first agent.
        :param agent_2_index: Index of the second agent.
        :return: None
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
                        if agent_1.policy_type == Policy.PPO:
                            action, log_prob, value = agent_1.policy.get_action(state, mask, deterministic=True)
                            past_log_prob_agent_1 = log_prob
                            past_value_agent_1 = value
                        elif agent_1.policy_type == Policy.DQN:
                            action = agent_1.policy.act(state=state, mask=mask, deterministic=True)
                        past_action_agent_1 = action
                        past_mask_agent_1 = mask
                        past_state_agent_1 = state

                        past_done_agent_1 = termination or truncation

                    elif agent == "player_1":
                        state = observation["observation"]
                        state = state.flatten()
                        if agent_2.policy_type == Policy.PPO:
                            action, log_prob, value = agent_2.policy.get_action(state, mask, deterministic=True)
                            past_log_prob_agent_2 = log_prob
                            past_value_agent_2 = value
                        elif agent_2.policy_type == Policy.DQN:
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


        return agent_1_win, agent_2_win, draws

    def fight_agent_against_random(self, num_fights: int = 2000,agent_id : int = 0) -> None:
        """
        Simulates fights between a player agent and a randomly selected opponent agent.

        :param num_fights: Number of fights to simulate. Defaults to 2000.
        :param agent_id: Index of the agent to fight.
        :return: None
        """
        # get a random agent
        random_agent: Agent = self.agents[agent_id]
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
                        if random_agent.policy_type == Policy.PPO or random_agent.policy_type == Policy.A2C:
                            random_agent.policy.experience_replay.add_step(state=past_state,
                                                                           action=past_action,
                                                                           reward=past_reward,
                                                                           next_state=next_state,
                                                                           done=past_done,
                                                                           old_log_prob=past_log_prob,
                                                                           value=past_value,
                                                                           action_mask=past_mask)
                        elif random_agent.policy_type == Policy.DQN:

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
                        if random_agent.policy_type == Policy.PPO or random_agent.policy_type == Policy.A2C:
                            action, log_prob, value = random_agent.policy.get_action(state, mask)
                            past_log_prob = log_prob
                            past_value = value
                        elif random_agent.policy_type == Policy.DQN:
                            action = random_agent.policy.act(state=state, mask=mask)

                        past_action = action
                        past_mask = mask
                        past_state = state

                        past_done = termination or truncation

                    else:
                        action = self.env.action_space(agent).sample(mask)
                self.env.step(action)
            if fight % 20 == 0 and random_agent.policy_type == Policy.DQN:
                random_agent.policy.update_epsilon()
            self.compute_winner(random_agent_win, opponent_win, draws, current_episode_reward,
                                current_episode_reward_opponent)
            random_agent.policy.writer.add_scalar("Reward", current_episode_reward, fight)
            self.env.close()

        # Calculating win rates over time
        random_agent_win, opponent_win, draws = self.compute_winrate_over_time(num_fights, random_agent_win,
                                                                               opponent_win, draws)






    def plot_winrate_over_time_1v1(self, agent_1: Agent, agent_2: Agent, agent_1_win: List[int], agent_2_win: List[int],
                                   draws: List[int]) -> None:
        """
        Plots the win rate over time.
        :param agent_1: The first agent.
        :param agent_2: The second agent.
        :param agent_1_win: List of wins for the first agent.
        :param agent_2_win: List of wins for the second agent.
        :param draws: List of draws.
        :return: None
        """
        plot_winrate_over_time(agent_1=agent_1, agent_1_win=agent_1_win,agent_2=agent_2, agent_2_win=agent_2_win, draws=draws)

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

    def test_agents_against_random(self, num_tests: int = 1000,agent_id : int = 0) -> None:
        """
        Tests the agents against each other.

        Parameters:
            num_tests (int): Number of tests to conduct. Defaults to 1000.
        """
        # get a random agent
        random_agent: Agent = self.agents[agent_id]
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
                        if random_agent.policy_type == Policy.PPO or random_agent.policy_type == Policy.A2C:
                            action, log_prob, value = random_agent.policy.get_action(state, mask, deterministic=True)
                            past_log_prob = log_prob
                            past_value = value
                        elif random_agent.policy_type == Policy.DQN:
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
        print(f"Win rate for agent {random_agent.id} against random agent: {random_agent_win[-1]}")
        #self.plot_winrate_over_time(random_agent, random_agent_win, opponent_win, draws)

        return action_list





texas_population = Population(connect_four_v3.env(), 1)
number_DQN = 5
number_PPO = 5
number_A2C = 1
texas_population.add_agent(Policy.PPO)
texas_population.test_agents_against_random(1000,0)
texas_population.fight_agent_against_random(50,0)
texas_population.test_agents_against_random(1000,0)
"""for i in range(number_DQN):
    texas_population.add_agent(Policy.DQN)
for i in range(number_PPO):
    texas_population.add_agent(Policy.PPO)
texas_population.training_loop(500)"""
# keep only the best 10 agent and the worst 10 agents (based on elo)
"""
texas_population.agents = sorted(texas_population.agents, key=lambda x: texas_population.rating.get_rating(x.id))
texas_population.agents = texas_population.agents[:10] + texas_population.agents[-10:]

action_list = []
elo_dict = {}
for index, agent in enumerate(texas_population.agents):
    action_list.append(texas_population.test_agents_against_random(100,agent_id=index))
    elo_dict[index] = texas_population.rating.get_rating(agent.id)
plot_strategy_landscape_elo_fading(action_list, [agent.policy_type for agent in texas_population.agents],agent_elo=elo_dict)

"""
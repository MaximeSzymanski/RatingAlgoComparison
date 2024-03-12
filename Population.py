from typing import List, Dict
from pettingzoo.utils.env import AECEnv
from utils.policy import Policy
from utils.logger import Logger
from rating.rating import TrueSkill, Elo
from  utils.diversity_action import Diversity
from models.DQN import DQN
from models.PPO import PPO
from models.A2C import A2C
from models.Random import Random
from models.Deterministic import Deterministic
from Agent import Agent
import numpy as np
from tqdm import tqdm
from pettingzoo.classic import connect_four_v3
class Population:
    def __init__(self, env: AECEnv, agent_counts : Dict[Policy, int],num_trials : int , num_rounds : int) -> None:
        """
        Initialize the population of agents.

        Parameters:
        - env (AECEnv): The environment.
        - agent_counts (Dict[Policy, int]): A dictionary containing the count of each agent type.
        """
        self.agents: List[Agent] = []
        self.env: AECEnv = env
        self.state_size = 84
        self.logger = Logger("logs")
        self.rating = Elo()
        self.base_rating = 1500
        self.action_size = env.action_space("player_1").n
        self.deterministic_action = list(range(self.action_size))
        self.agent_counts = agent_counts
        self.num_trials = num_trials
        self.num_rounds = num_rounds
        self.reset_population()
        self.diversity = Diversity(num_trials=num_trials, num_rounds=num_rounds, num_agents=len(self.agents), agents=self.agents)



    def get_agent_type_per_id(self, id: int) -> Policy:
        """
        Get the type of agent for a given id.

        Parameters:
        - id (int): The id of the agent.

        Returns:
        - Policy: The type of agent.
        """
        return self.agents[id].policy_type

    def get_dict_agent_type_per_id(self) -> Dict[int, Policy]:
        """
        Get the type of agent for each id.

        Returns:
        - Dict[int, Policy]: The type of agent for each id.
        """
        return {agent.id: agent.policy_type for agent in self.agents}

    def get_id_new_agent(self) -> int:
        """
        Get the id for a new agent.

        Returns:
        - int: The id for a new agent.
        """
        return len(self.agents)

    def add_agent(self, policy_type: Policy) -> None:
        """
        Add an agent to the population.

        Parameters:
        - policy_type (Policy): The type of policy to use for the agent.
        """
        if policy_type == Policy.Deterministic:
            for action in self.deterministic_action:
                id = self.get_id_new_agent()
                self.agents.append(Agent(policy_type, self.state_size, self.action_size, id=id,
                                         action_deterministic=action, agent_index=self.number_agents_per_algo[policy_type]))
                self.rating.add_player(id)
        else:
            id = self.get_id_new_agent()
            self.agents.append(Agent(policy_type, self.state_size, self.action_size,
                                     id=id, agent_index=self.number_agents_per_algo[policy_type]))
            self.rating.add_player(id)
            self.number_agents_per_algo[policy_type] += 1
        self.agents.sort(key=self.policy_sort)
        # reassing the id of the agents based on the new order
        for i, agent in enumerate(self.agents):
            agent.id = i
            self.rating.update_id(new_id=i, old_id=agent.id)


    def build_population(self) -> None:
        """
        Build the population of agents.
        """

        for policy, count in self.agent_counts.items():
            for _ in range(count):
                self.add_agent(policy)

    def reset_population(self) -> None:
        """
        Reset the population of agents.
        """
        self.agents = []
        self.rating = Elo()
        self.number_agents_per_algo = {policy: 0 for policy in Policy}
        self.build_population()



    def policy_sort(self, agent: Agent) -> float:
        """
        Sort agents based on their policy type.

        Parameters:
        - agent (Agent): The agent to sort.

        Returns:
        - float: The sorting key.
        """
        if agent.policy_type == Policy.Deterministic or agent.policy_type == Policy.Random:
            return 'zzz'
        else:
            return agent.policy_type.name

    # Add other methods here...

    def generate_random_states_and_masks(self, num_states: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generates random states and masks.

        Parameters:
            num_states (int): Number of states to generate. Defaults to 1000.

        Returns:
            Tuple of lists containing random states and masks.
        """
        states = []
        masks = []
        while len(states) < num_states:
            state = self.env.observation_space("player_1").sample()
            mask = state["action_mask"]
            if mask.sum() == 0:
                continue
            state = state["observation"].flatten()
            states.append(state)
            masks.append(mask)

        return states, masks

    def remove_agent(self, agent: Agent) -> None:
        """
        Remove an agent from the population.

        Parameters:
        - agent (Agent): The agent to remove.
        """
        self.rating.remove_player(agent.id)
        self.agents.remove(agent)

    def update_diversity_matrix(self, num_trial: int, num_round: int,num_states_diversity : int) -> None:
        """
        Log the diversity matrix.

        Parameters:
        - num_trial (int): The number of the trial.
        - num_round (int): The number of the round.
        """
        states, masks = self.generate_random_states_and_masks(num_states_diversity)

        self.diversity.calculate_diversity_round(num_trial=num_trial, num_round=num_round, agents=self.agents, states=states,
                                                 masks=masks)
        self.diversity.build_diversity_matrix_round(num_trial=num_trial, num_round=num_round)
        if num_round > 1:
            self.logger.log_diversity_matrix(diversity_matrix=self.diversity.get_diversity_matrix(num_trial=num_trial, num_round=num_round),
                                                num_trial=num_trial, num_round=num_round, agents=self.agents)
            self.logger.log_diversity_per_policy_trial_until_round(diversity_per_agent=self.diversity.get_diversity_per_type_of_policy_until_round_specific_trial(num_round=num_round,num_trial=num_trial),
                                                                    num_trial=num_trial, num_round=num_round)



    def training_loop(self,  num_fights_train=10, num_fight_test=10,use_rating_in_reward=False,num_states_diversity : int  = 1000) -> None:
        """
        The training loop for the population.

        Parameters:
        - number_round (int): The number of rounds to train for.
        - num_fights_train (int): The number of fights to train for.
        - num_fight_test (int): The number of fights to test for.
        - use_rating_in_reward (bool): Whether to use the rating in the reward.
        - num_trial (int): The number of trials to perform. Defaults to 1.
        """

        num_non_random_deterministic_agents = len([agent for agent in self.agents if agent.policy_type != Policy.Random and agent.policy_type != Policy.Deterministic])

        policy_names = set([agent.policy_name for agent in self.agents])
        for trial in range(self.num_trials):
            rating_per_policy_mean = {policy: [] for policy in policy_names}
            rating_per_policy_std = {policy: [] for policy in policy_names}

            for round in tqdm(range(self.num_rounds)):
                self.update_diversity_matrix(num_round=round,num_trial= trial,num_states_diversity=num_states_diversity)

                rating_per_policy_mean_round = {policy: [] for policy in policy_names}
                paired_agents = self.rating.find_similar_rating_pairs()
                for agent_1, agent_2 in paired_agents:
                    _, _, _ = self.train_fight_1vs1(agent_1_index=agent_1, agent_2_index=agent_2,
                                                    num_fights=num_fights_train, use_rating_in_reward=use_rating_in_reward)
                    agent_1_win, agent_2_win, _ = self.test_fight_1vs1(num_fights=num_fight_test, agent_1_index=agent_1,
                                                                           agent_2_index=agent_2)
                    agent_1_win = sum(agent_1_win)
                    agent_2_win = sum(agent_2_win)
                    draws = False
                    if agent_1_win > agent_2_win:
                        winner = agent_1
                        loser = agent_2
                    elif agent_1_win < agent_2_win:
                        winner = agent_2
                        loser = agent_1
                    else:
                        draws = True
                    self.rating.update_ratings(winner, loser, draw=draws)
                for agent in self.agents:
                    for policy in policy_names:
                        if agent.policy_name == policy:
                            rating_per_policy_mean_round[policy].append(self.rating.get_rating(agent.id, to_plot=False))
                for policy in policy_names:
                    rating_per_policy_mean[policy].append((rating_per_policy_mean_round[policy]))
                    rating_per_policy_std[policy].append((rating_per_policy_mean_round[policy]))
            self.logger.plot_rating_per_policy(policies=policy_names, rating_mean=rating_per_policy_mean,
                                               rating_std=rating_per_policy_std, num_trial=trial, rating=self.rating)
            self.reset_population()
        self.logger.log_diversity_per_type_of_policy_averaged_over_trials(diversity_per_type=self.diversity.get_diversity_per_type_of_policy_all_trial(num_round=self.num_rounds-1, num_trials=self.num_trials-1))
    def train_fight_1vs1(self, num_fights: int = 1000, agent_1_index: int = 0, agent_2_index: int = 1, use_rating_in_reward=False) -> tuple[
        List[int], List[int], List[int]]:
        """
        Simulates fights between two agents.

        Parameters:
        - num_fights (int): Number of fights to simulate.
        - agent_1_index (int): Index of the first agent.
        - agent_2_index (int): Index of the second agent.
        - use_rating_in_reward (bool): Whether to use the rating in the reward.
        """
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
        reward_rating_factor_agent_1 = 1
        reward_rating_factor_agent_2 = 1
        if use_rating_in_reward:
            rating_diff = self.rating.get_rating(agent_1.id) - self.rating.get_rating(agent_2.id)
            rating_factor = 1 / (np.exp(-abs(rating_diff) / 400))
            if rating_diff < 0:
                reward_rating_factor_agent_1 = (rating_factor)
                reward_rating_factor_agent_2 = abs(2-rating_factor)
            elif rating_diff > 0:
                reward_rating_factor_agent_1 = abs(2-rating_factor)
                reward_rating_factor_agent_2 = (rating_factor)
        for fight in range(num_fights):
            self.env.reset(seed=42)
            current_episode_reward_agent_1 = 0
            current_episode_reward_agent_2 = 0
            draw_count = 0
            step = 0
            update_freq = 4
            for agent in self.env.agent_iter():
                if agent == "player_0" and agent_1.policy.experience_replay.can_train() and step % update_freq == 0:
                    agent_1.policy.train_agent()
                    pass
                elif agent == "player_1" and agent_2.policy.experience_replay.can_train() and step % update_freq == 0:
                    agent_2.policy.train_agent()
                    pass
                observation, reward, termination, truncation, info = self.env.last()
                if agent == "player_0":
                    if past_state_agent_1 is not None and past_action_agent_1 is not None and past_reward_agent_1 is not None and past_mask_agent_1 is not None:
                        next_state = observation["observation"]
                        next_state = next_state.flatten()
                        if agent_1.policy_type == Policy.PPO or agent_1.policy_type == Policy.A2C:
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
                    if use_rating_in_reward and reward != 0:
                        if reward == 1:
                            reward = reward * reward_rating_factor_agent_1
                        elif reward == -1:
                            reward = reward * reward_rating_factor_agent_2
                    past_reward_agent_1 = reward
                    current_episode_reward_agent_1 += reward
                elif agent == "player_1":
                    if past_state_agent_2 is not None and past_action_agent_2 is not None and past_reward_agent_2 is not None and past_mask_agent_2 is not None:
                        next_state = observation["observation"]
                        next_state = next_state.flatten()
                        if agent_2.policy_type == Policy.PPO or agent_2.policy_type == Policy.A2C:
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
                    if use_rating_in_reward and reward != 0:
                        if reward == 1:
                            reward = reward * reward_rating_factor_agent_2
                        elif reward == -1:
                            reward = reward * reward_rating_factor_agent_1
                    past_reward_agent_2 = reward
                    current_episode_reward_agent_2 += reward
                if termination or truncation:
                    action = None
                else:
                    mask = observation["action_mask"]
                    if agent == "player_0":
                        state = observation["observation"]
                        state = state.flatten()
                        if agent_1.policy_type == Policy.PPO or agent_1.policy_type == Policy.A2C:
                            action, log_prob, value = agent_1.policy.get_action(
                                state, mask)
                            past_log_prob_agent_1 = log_prob
                            past_value_agent_1 = value
                        elif agent_1.policy_type == Policy.DQN or agent_1.policy_type == Policy.Random or agent_1.policy_type == Policy.Deterministic:
                            action = agent_1.policy.act(state=state, mask=mask)
                        past_action_agent_1 = action
                        past_mask_agent_1 = mask
                        past_state_agent_1 = state
                        past_done_agent_1 = termination or truncation
                    elif agent == "player_1":
                        state = observation["observation"]
                        state = state.flatten()
                        if agent_2.policy_type == Policy.PPO or agent_2.policy_type == Policy.A2C:
                            action, log_prob, value = agent_2.policy.get_action(
                                state, mask)
                            past_log_prob_agent_2 = log_prob
                            past_value_agent_2 = value
                        elif agent_2.policy_type == Policy.DQN or agent_2.policy_type == Policy.Random or agent_2.policy_type == Policy.Deterministic:
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
            agent_1.policy.writer.add_scalar(
                "Reward", current_episode_reward_agent_1, agent_1.num_fights)
            agent_2.policy.writer.add_scalar(
                "Reward", current_episode_reward_agent_2, agent_2.num_fights)
            agent_1.num_fights += 1
            agent_2.num_fights += 1
            self.env.close()
        agent_1_win, agent_2_win, draws = self.compute_winrate_over_time(num_fights, agent_1_win, agent_2_win, draws)
        return agent_1_win, agent_2_win, draws

    def test_fight_1vs1(self, num_fights: int = 2000, agent_1_index: int = 0, agent_2_index: int = 1) -> tuple[
        List[int], List[int], List[int]]:
        """
        Simulates fights between two agents, without training in a deterministic manner.

        Parameters:
            num_fights (int): Number of fights to simulate. Defaults to 2000.
            agent_1_index (int): Index of the first agent.
            agent_2_index (int): Index of the second agent.

        Returns:
            Tuple containing lists of wins for each agent and draws.
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
                        if agent_1.policy_type == Policy.PPO or agent_1.policy_type == Policy.A2C:
                            action, log_prob, value = agent_1.policy.get_action(
                                state, mask, deterministic=True)
                            past_log_prob_agent_1 = log_prob
                            past_value_agent_1 = value
                        elif agent_1.policy_type == Policy.DQN or agent_1.policy_type == Policy.Random or agent_1.policy_type == Policy.Deterministic:
                            action = agent_1.policy.act(
                                state=state, mask=mask, deterministic=True)
                        past_action_agent_1 = action
                        past_mask_agent_1 = mask
                        past_state_agent_1 = state

                        past_done_agent_1 = termination or truncation

                    elif agent == "player_1":
                        state = observation["observation"]
                        state = state.flatten()
                        if agent_2.policy_type == Policy.PPO or agent_2.policy_type == Policy.A2C:
                            action, log_prob, value = agent_2.policy.get_action(
                                state, mask, deterministic=True)
                            past_log_prob_agent_2 = log_prob
                            past_value_agent_2 = value
                        elif agent_2.policy_type == Policy.DQN or agent_2.policy_type == Policy.Random or agent_2.policy_type == Policy.Deterministic:
                            action = agent_2.policy.act(
                                state=state, mask=mask, deterministic=True)
                        past_action_agent_2 = action
                        past_mask_agent_2 = mask
                        past_state_agent_2 = state

                        past_done_agent_2 = termination or truncation

                self.env.step(action)

            self.compute_winner(agent_1_win, agent_2_win, draws, current_episode_reward_agent_1,
                                current_episode_reward_agent_2)

            self.env.close()

        # Calculating win rates over time
        agent_1_win, agent_2_win, draws = self.compute_winrate_over_time(
            num_fights, agent_1_win, agent_2_win, draws)

        return agent_1_win, agent_2_win, draws

    def fight_agent_against_random(self, num_fights: int = 2000, agent_id: int = 0) -> None:
        """
        Simulates fights between a player agent and a randomly selected opponent agent.

        Parameters:
            num_fights (int): Number of fights to simulate. Defaults to 2000.
            agent_id (int): Index of the agent to fight.
        """
        #get a random agent
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
                            action, log_prob, value = random_agent.policy.get_action(
                                state, mask)
                            past_log_prob = log_prob
                            past_value = value
                        elif random_agent.policy_type == Policy.DQN or random_agent.policy_type == Policy.Random or random_agent.policy_type == Policy.Deterministic:
                            action = random_agent.policy.act(
                                state=state, mask=mask)

                        past_action = action
                        past_mask = mask
                        past_state = state

                        past_done = termination or truncation

                    else:
                        action = self.env.action_space(agent).sample(mask)
                self.env.step(action)
            if fight % 100 == 0 and random_agent.policy_type == Policy.DQN:
                random_agent.policy.update_epsilon()
            self.compute_winner(random_agent_win, opponent_win, draws, current_episode_reward,
                                current_episode_reward_opponent)
            random_agent.policy.writer.add_scalar(
                "Reward", current_episode_reward, random_agent.num_fights)
            self.env.close()

        # Calculating win rates over time
        random_agent_win, opponent_win, draws = self.compute_winrate_over_time(num_fights, random_agent_win,
                                                                               opponent_win, draws)

    def generate_random_states_and_masks(self, num_states: int = 1000) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Generates random states and masks.

        Parameters:
            num_states (int): Number of states to generate. Defaults to 1000.

        Returns:
            Tuple of lists containing random states and masks.
        """


        states = []
        masks = []
        while len(states) < num_states:
            state = self.env.observation_space("player_1").sample()
            mask = state["action_mask"]
            if mask.sum() == 0:
                continue
            state = state["observation"].flatten()
            states.append(state)
            masks.append(mask)
        return states, masks

    def plot_winrate_over_time_1v1(self, agent_1: Agent, agent_2: Agent, agent_1_win: List[int], agent_2_win: List[int],
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
        random_agent_win = np.cumsum(
            random_agent_win) / (np.arange(num_fights) + 1)
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

    def test_agents_against_random(self, num_tests: int = 1000, agent_id: int = 0) -> None:
        """
        Tests the agents against each other.

        Parameters:
            num_tests (int): Number of tests to conduct. Defaults to 1000.
            agent_id (int): Index of the agent to fight.
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
                            action, log_prob, value = random_agent.policy.get_action(
                                state, mask, deterministic=True)
                            past_log_prob = log_prob
                            past_value = value
                        elif random_agent.policy_type == Policy.DQN or random_agent.policy_type == Policy.Random or random_agent.policy_type == Policy.Deterministic:
                            action = random_agent.policy.act(
                                state=state, mask=mask, deterministic=True)
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

        print(
            f"Win rate for agent {random_agent.id} against random agent: {random_agent_win[-1]}")
        return action_list



agent_counts = {
    Policy.DQN: 1,
    Policy.PPO: 1,
    Policy.A2C: 1,
    Policy.Random: 1,
    Policy.Deterministic:1
}
texas_population = Population(connect_four_v3.env(),agent_counts, num_trials=5, num_rounds=100)
num_fights_train = 100
num_fight_test = 5
texas_population.training_loop(num_fights_train=num_fights_train,
                               num_fight_test=num_fight_test,use_rating_in_reward=False)


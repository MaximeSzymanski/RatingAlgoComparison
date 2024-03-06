from Agent import Agent
import numpy as np
from models.PPO import PPO
from models.DQN import DQN
from pettingzoo.utils.env import AECEnv
from pettingzoo.classic import connect_four_v3,tictactoe_v3
import matplotlib.pyplot as plt
from typing import List



class Population():
    def __init__(self, policy_type, env : AECEnv, num_agents) -> None:
        self.agents : list[Agent] = []
        self.env : AECEnv = env
        self.state_size = 84
        
        self.action_size = env.action_space("player_1").n
        for i in range(num_agents):

            self.agents.append(Agent(policy_type, self.state_size, self.action_size))


    #def fight_1vs1(self,num_fights,agent_1_index, agent_2_index):


    def fight_agent_against_random(self, num_fights : int =2000) -> None:
        """
        Simulates fights between a player agent and a randomly selected opponent agent.

        Parameters:
            num_fights (int): Number of fights to simulate. Defaults to 2000.
        """
        # get a random agent
        random_agent : Agent = self.agents[np.random.randint(len(self.agents))]
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
                           
                            random_agent.policy.experience_replay.add_step(state=past_state,action=past_action,reward=past_reward,
                                                                next_state=next_state,done=past_done,action_mask=past_mask)
                            

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
                            action, log_prob , value = random_agent.policy.get_action(state, mask)
                            past_log_prob = log_prob
                            past_value = value
                        elif random_agent.policy_type == "DQN":
                            action = random_agent.policy.act(state=state,mask=mask)
                        past_action = action
                        past_mask = mask
                        past_state = state

                        
                        
                        past_done = termination or truncation

                    else:
                        action = self.env.action_space(agent).sample(mask)
                        
                self.env.step(action)
            if fight % 20 == 0:
                random_agent.policy.update_epsilon()
            self.compute_winner(random_agent_win, opponent_win, draws, current_episode_reward, current_episode_reward_opponent)
            random_agent.policy.writer.add_scalar("Reward",current_episode_reward,fight)
            self.env.close()
            
        # Calculating win rates over time
        random_agent_win, opponent_win, draws = self.compute_winrate_over_time(num_fights, random_agent_win, opponent_win, draws)
        
        # Plotting win rates over time
        self.plot_winrate_over_time(random_agent, random_agent_win, opponent_win, draws)
        self.test_agents()

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
        random_agent : Agent = self.agents[np.random.randint(len(self.agents))]
        print("Random agent rating: ", random_agent.rating)
        random_agent_win = []
        opponent_win = []
        draws = []

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
                            action, log_prob , value = random_agent.policy.get_action(state, mask,deterministic=True)
                            past_log_prob = log_prob
                            past_value = value
                        elif random_agent.policy_type == "DQN":
                            action = random_agent.policy.act(state=state,mask=mask,deterministic=True)
                        past_action = action
                        past_mask = mask
                        past_state = state
                       
                        past_done = termination or truncation
                        current_episode_reward += reward

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
        random_agent_win = np.cumsum(random_agent_win) / (np.arange(num_tests) + 1)
        opponent_win = np.cumsum(opponent_win) / (np.arange(num_tests) + 1)
        draws = np.cumsum(draws) / (np.arange(num_tests) + 1)
        
        # Plotting win rates over time
        plt.plot(random_agent_win, label=f"{random_agent.policy_type} Win Rate")
        plt.plot(opponent_win, label="Opponent Win Rate")
        plt.plot(draws, label="Draw Rate")
        plt.xlabel("Number of Fights")
        plt.ylabel("Win Rate")
        plt.legend()
        plt.title("Win Rate Over Time")
        plt.show()
texas_population = Population("DQN", connect_four_v3.env(), 1)
texas_population.fight_agent_against_random()
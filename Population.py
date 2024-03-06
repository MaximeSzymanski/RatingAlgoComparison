from Agent import Agent
import numpy as np
from models.PPO import PPO
from pettingzoo.utils.env import AECEnv
from pettingzoo.classic import connect_four_v3,tictactoe_v3
import matplotlib.pyplot as plt
class Population():
    def __init__(self, policy_type, env : AECEnv, num_agents) -> None:
        self.agents : list[Agent] = []
        self.env : AECEnv = env
        self.state_size = 18
        
        self.action_size = env.action_space("player_1").n
        for i in range(num_agents):

            self.agents.append(Agent(policy_type, self.state_size, self.action_size))


    def fight_agent_against_random(self, num_fights=10000):
        # get a random agent
        random_agent : Agent = self.agents[np.random.randint(len(self.agents))]
        print("Random agent rating: ", random_agent.rating)
        random_agent_win = []
        opponent_win = []
        draws = []

        for fight in range(num_fights):
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
                if agent == "player_1" and random_agent.policy.experience_replay.can_train():
                    print("Training agent", "episode: ", fight, "reward: ", current_episode_reward)
                    random_agent.policy.train_agent()
                    
                observation, reward, termination, truncation, info = self.env.last()
                
                # check if the agent is the random agent
                if agent == "player_1":
                    if past_state is not None and past_action is not None and past_reward is not None and past_mask is not None:
                        next_state = observation["observation"]
                        # flatten
                        next_state = next_state.flatten()
                        
                        random_agent.policy.experience_replay.add_step(state=past_state,
                                                                    action=past_action,
                                                                    reward=past_reward,
                                                                    next_state=next_state,
                                                                    done=past_done,
                                                                    old_log_prob=past_log_prob,
                                                                    value=past_value,
                                                                    action_mask=past_mask)

                    past_state = observation["observation"]
                    past_reward = reward
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
                        action, log_prob , value = random_agent.policy.get_action(state, mask)
                        past_action = action
                        past_mask = mask
                        past_state = state
                        past_value = value
                        past_log_prob = log_prob
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
        random_agent_win = np.cumsum(random_agent_win) / (np.arange(num_fights) + 1)
        opponent_win = np.cumsum(opponent_win) / (np.arange(num_fights) + 1)
        draws = np.cumsum(draws) / (np.arange(num_fights) + 1)
        
        # Plotting win rates over time
        plt.plot(random_agent_win, label="Random Agent Win Rate")
        plt.plot(opponent_win, label="Opponent Win Rate")
        plt.plot(draws, label="Draw Rate")
        plt.xlabel("Number of Fights")
        plt.ylabel("Win Rate")
        plt.legend()
        plt.title("Win Rate Over Time")
        plt.show()
        self.test_agents()

    def test_agents(self, num_tests=1000):
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
                        action, log_prob , value = random_agent.policy.get_action(state, mask,deterministic=True)
                        past_action = action
                        past_mask = mask
                        past_state = state
                        past_value = value
                        past_log_prob = log_prob
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
        plt.plot(random_agent_win, label="Random Agent Win Rate")
        plt.plot(opponent_win, label="Opponent Win Rate")
        plt.plot(draws, label="Draw Rate")
        plt.xlabel("Number of Fights")
        plt.ylabel("Win Rate")
        plt.legend()
        plt.title("Win Rate Over Time")
        plt.show()
texas_population = Population("PPO", tictactoe_v3.env(), 1)
texas_population.fight_agent_against_random()
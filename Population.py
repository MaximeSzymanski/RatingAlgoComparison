from Agent import Agent
import numpy as np
from models.PPO import PPO
from pettingzoo.utils.env import AECEnv
from pettingzoo.classic import connect_four_v3
import matplotlib.pyplot as plt
class Population():
    def __init__(self, policy_type, env : AECEnv, num_agents) -> None:
        self.agents = []
        self.env = env
        self.state_size = 84
        self.action_size = env.action_space("player_0").n
        for i in range(num_agents):

            self.agents.append(Agent(policy_type, self.state_size, self.action_size))


    def fight_agent_against_random(self, num_fights = 300000):
        # get a random agent
        random_agent = self.agents[np.random.randint(len(self.agents))]
        print("Random agent rating: ", random_agent.rating)
        random_agent_win = []
        opponent_win = []

        for fight in range(num_fights):
            # our agent is player 1 and the random bot is player 2
            past_state = None
            past_action = None
            past_reward = None
            past_mask = None
            self.env.reset(seed=42)
            current_episode_reward = 0
            current_episode_reward_opponent = 0
            for agent in self.env.agent_iter():
                # chek if the agent can train
                if agent == "player_0" and random_agent.policy.experience_replay.can_train():
                    print("Training agent", "episode: ", fight, "reward: ", current_episode_reward)
                    random_agent.policy.train_agent()
                observation, reward, termination, truncation, info = self.env.last()
                # check if the agent is the random agent
                if agent == "player_0":
                    if past_state is not None and past_action is not None and past_reward is not None and past_mask is not None:
                        current_episode_reward += reward
                        next_state = observation["observation"]
                        # flatten
                        next_state = next_state.flatten()
                        _, log_prob, value = random_agent.policy.get_action(past_state,past_mask )
                        random_agent.policy.experience_replay.add_step(state=past_state,
                                                                       action=past_action,
                                                                       reward=past_reward,
                                                                       next_state=next_state,
                                                                       done=termination,
                                                                       old_log_prob=log_prob,
                                                                       value=value,
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
                    if agent == "player_0":
                        state = observation["observation"]
                        state = state.flatten()
                        action ,_ ,_ = random_agent.policy.get_action(state, mask)
                        past_action = action
                        past_mask = mask
                        past_state = state
                    else :
                        action = self.env.action_space(agent).sample(mask)
                self.env.step(action)
            random_agent_win.append(current_episode_reward)
            opponent_win.append(current_episode_reward_opponent)
            self.env.close()
        # plot reward over time, averaged each 100 episodes
        random_agent_win = np.array(random_agent_win)
        random_agent_win = random_agent_win.reshape(-1,100)
        random_agent_win = np.mean(random_agent_win, axis=1)

        opponent_win = np.array(opponent_win)
        opponent_win = opponent_win.reshape(-1,100)
        opponent_win = np.mean(opponent_win, axis=1)
        plt.plot(opponent_win, label="opponent")


        plt.plot(random_agent_win, label="random agent")
        plt.legend()
        plt.show()



texas_population = Population("PPO", connect_four_v3.env(), 1)
texas_population.fight_agent_against_random()
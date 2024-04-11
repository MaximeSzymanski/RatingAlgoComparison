from popcore import Interaction
from utils.policy import Policy
import os
import multiprocessing
from pettingzoo.classic import connect_four_v3
from models.DQN import DQN
from models.PPO import PPO
from models.A2C import A2C
from models.Random import Random
import json
import datetime

def compute_winner(agent_1_win, agent_2_win, draws, current_episode_reward_agent_1 : int, current_episode_reward_agent_2: int) -> None:
        """
        Computes the winner of a fight.

        Parameters:
            random_agent_win (List[int]): List of wins for the random agent.
            opponent_win (List[int]): List of wins for the opponent agent.
            draws (List[int]): List of draws.
            current_episode_reward (int): Reward obtained by the player agent.
            current_episode_reward_opponent (int): Reward obtained by the opponent agent.
        """
        if current_episode_reward_agent_1 > current_episode_reward_agent_2:
            agent_1_win.append(1)
            agent_2_win.append(0)
            draws.append(0)
            
        elif current_episode_reward_agent_1 < current_episode_reward_agent_2:
            agent_1_win.append(0)
            agent_2_win.append(1)
            draws.append(0)
            
        else:
            agent_1_win.append(0)
            agent_2_win.append(0)
            draws.append(1)

def test_fight_1vs1(env, agent_1, agent_2, num_fights, agent_1_index: int = 0, agent_2_index: int = 1):
    # TODO: random opening ?
    # This code better be correct because I sure as hell ain't checking

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

    interactions = []

    for fight in range(num_fights):
        # our agent is player 1 and the random bot is player 2

        env.reset(seed=42)
        current_episode_reward_agent_1 = 0
        current_episode_reward_agent_2 = 0
        draw_count = 0
        step = 0
        update_freq = 4
        for agent in env.agent_iter():
            # check if the agent can train

            observation, reward, termination, truncation, info = env.last()

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
                    if agent_1.policy_type == "PPO" or agent_1.policy_type == "A2C":
                        action, log_prob, value = agent_1.get_action(
                            state, mask, deterministic=True)
                        past_log_prob_agent_1 = log_prob
                        past_value_agent_1 = value
                    elif agent_1.policy_type == "DQN" or agent_1.policy_type == "Random" or agent_1.policy_type == "Deterministic":
                        action = agent_1.act(
                            state=state, mask=mask, deterministic=True)
                    past_action_agent_1 = action
                    past_mask_agent_1 = mask
                    past_state_agent_1 = state

                    past_done_agent_1 = termination or truncation

                elif agent == "player_1":
                    state = observation["observation"]
                    state = state.flatten()
                    if agent_2.policy_type == "PPO" or agent_2.policy_type == "A2C":
                        action, log_prob, value = agent_2.get_action(
                            state, mask, deterministic=True)
                        past_log_prob_agent_2 = log_prob
                        past_value_agent_2 = value
                    elif agent_2.policy_type == "DQN" or agent_2.policy_type =="Random" or agent_2.policy_type == "Deterministic":
                        action = agent_2.act(
                            state=state, mask=mask, deterministic=True)
                    past_action_agent_2 = action
                    past_mask_agent_2 = mask
                    past_state_agent_2 = state

                    past_done_agent_2 = termination or truncation

            env.step(action)

        compute_winner(agent_1_win, agent_2_win, draws, current_episode_reward_agent_1, current_episode_reward_agent_2)
        for i, _ in enumerate(agent_1_win):
            result = [agent_1_win[i], agent_2_win[i]] if not draws[i] else [0.5, 0.5]
            interactions.append(Interaction([agent_1_index, agent_2_index], result))

        env.close()

    return interactions

agents_ids = {}

def load_population(path, type, state_size, action_size):
    agents = []
    if type == "A2C":
        for agent_file in os.listdir(path):
            agent_name = os.path.join(path, agent_file)
            if agent_name not in agents_ids:
                agents_ids[agent_name] = len(agents_ids)
            id = agents_ids[agent_name]
            agent = A2C(state_size=state_size, action_size=action_size, num_steps=5, env_name='connect_four_v3', agent_index=id)
            agent.load_model(agent_name)
            agents.append((agent, agent_name))

    elif type == "DQN":
        for agent_file in os.listdir(path):
            agent_name = os.path.join(path, agent_file)
            if agent_name not in agents_ids:
                agents_ids[agent_name] = len(agents_ids)
            id = agents_ids[agent_name]
            agent = DQN(memory_size=10000, batch_size=64, gamma=0.99, epsilon=1, epsilon_decay=0.98,
                        epsilon_min=0.01, state_size=state_size, action_size=action_size, seed=42,
                        env_name="connect_four_v3", lr=3e-4, agent_index=id)
            agent.load_model(agent_name)
            agents.append((agent, agent_name))

    elif type == "PPO":
        for agent_file in os.listdir(path):
            agent_name = os.path.join(path, agent_file)
            if agent_name not in agents_ids:
                agents_ids[agent_name] = len(agents_ids)
            id = agents_ids[agent_name]
            agent = PPO(state_size=state_size, action_size=action_size, num_steps=2048,
                        batch_size=64, env_name='connect_four_v3', agent_index=id)
            agent.load_model(agent_name)
            agents.append((agent, agent_name))

    elif type == "random":
        for i in range(5):
            agent_name = f"random{i}"
            if agent_name not in agents_ids:
                agents_ids[agent_name] = len(agents_ids)
            id = agents_ids[agent_name]
            agent = Random(action_size=action_size, seed=0)
            agents.append((agent, agent_name))

    return agents

def partial_tournament(path, subpath):
    interactions = []
    env = connect_four_v3.env()
    state_size = 84
    action_size = env.action_space("player_1").n
    for subsubpath in os.listdir(os.path.join(path, subpath)):
        pop1 = load_population(os.path.join(path, subpath, subsubpath), subpath, state_size, action_size)

        for subpath2 in os.listdir(path):
            for subsubpath2 in os.listdir(os.path.join(path, subpath2)):
                pop2 = load_population(os.path.join(path, subpath2, subsubpath2), subpath2, state_size, action_size)
                print(os.path.join(path, subpath, subsubpath) + " vs. " + os.path.join(path, subpath2, subsubpath2))
                start = datetime.datetime.now()

                for agent_1, path_agent1 in pop1:
                    for agent_2, path_agent2 in pop2:
                        interactions.extend(test_fight_1vs1(env, agent_1, agent_2, 1, agents_ids[path_agent1], agents_ids[path_agent2]))
                print(datetime.datetime.now() - start)
    return interactions

def main():
    main_path = "fucked_up_data/saved_models"
    subpaths = os.listdir(main_path)

    #results = [partial_tournament(main_path, subpath) for subpath in subpaths]

    with multiprocessing.Pool() as pool:
        results = pool.starmap(partial_tournament, [(main_path, subpath) for subpath in subpaths])

    json_data = {"ids" : agents_ids, "interactions" : []}
    for interacs in results:
        for i in interacs:
            json_data["interactions"].append(
                {"p0": i.players[0],
                "p1": i.players[1],
                "score0": i.outcomes[0],
                "score1": i.outcomes[1]})
    
    with open("tournament_data.json", "w") as f:
        f.write(json.dumps(json_data))

if __name__ == "__main__":
    start = datetime.datetime.now()
    main()
    print(f"total {datetime.datetime.now() - start}")
import multiprocessing
from utils.policy import Policy
from Population import Population
from pettingzoo.classic import connect_four_v3

import datetime

# Hyperparameters

n_trials = 1
n_rounds = 40
n_opponents_per_agent = 20 #40
num_fights_train = 1 # that's the number of fight against *each* opponent in training
num_fight_test = 0
n_agents = 15 #15

def train_populationA2C(rating_system, rating_systems):
    print(rating_system)

    # train A2C
    agent_counts = {
        Policy.DQN: 0,
        Policy.PPO: 0,
        Policy.A2C: n_agents,
        Policy.Random: 0,
        Policy.Deterministic:0
    }
    exp_name = f"A2C/using_{rating_system}_for_matchmaking" # CHANGE THIS IF NOT USING DQN
    texas_population = Population(connect_four_v3.env(), agent_counts, num_trials=n_trials, num_rounds=n_rounds, num_opponnent_per_agent=n_opponents_per_agent, rating_system=rating_system, rating_systems=rating_systems, experiment_name=exp_name)
    texas_population.training_loop(num_fights_train=num_fights_train, use_rating_in_reward=False)

def train_populationDQN(rating_system, rating_systems):
    print(rating_system)
    agent_counts = {
            Policy.DQN: n_agents,
            Policy.PPO: 0,
            Policy.A2C: 0,
            Policy.Random: 0,
            Policy.Deterministic:0
        }
    exp_name = f"DQN/using_{rating_system}_for_matchmaking" # CHANGE THIS IF NOT USING DQN
    texas_population = Population(connect_four_v3.env(), agent_counts, num_trials=n_trials, num_rounds=n_rounds, num_opponnent_per_agent=n_opponents_per_agent, rating_system=rating_system, rating_systems=rating_systems, experiment_name=exp_name)
    texas_population.training_loop(num_fights_train=num_fights_train, use_rating_in_reward=False)

def train_populationPPO(rating_system, rating_systems):
    print(rating_system)
    # train PPO
    agent_counts = {
        Policy.DQN: 0,
        Policy.PPO: n_agents,
        Policy.A2C: 0,
        Policy.Random: 0,
        Policy.Deterministic:0
    }
    exp_name = f"PPO/using_{rating_system}_for_matchmaking" # CHANGE THIS IF NOT USING DQN
    texas_population = Population(connect_four_v3.env(), agent_counts, num_trials=n_trials, num_rounds=n_rounds, num_opponnent_per_agent=n_opponents_per_agent, rating_system=rating_system, rating_systems=rating_systems, experiment_name=exp_name)
    texas_population.training_loop(num_fights_train=num_fights_train, use_rating_in_reward=False)

if __name__ == '__main__':
    start = datetime.datetime.now()
    processes = []
    rating_systems = ["elo", "bayeselo", "glicko", "glicko2", "trueskill", "melo", "uniform"]
    for rating_system in rating_systems:
        process = multiprocessing.Process(target=train_populationA2C, args=(rating_system,rating_systems))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    
    for rating_system in rating_systems:
        process = multiprocessing.Process(target=train_populationPPO, args=(rating_system,rating_systems))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    for rating_system in rating_systems:
        process = multiprocessing.Process(target=train_populationDQN, args=(rating_system,rating_systems))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    
    print(f"Total experiment duration : {datetime.datetime.now() - start}")


# TODO: Train every type of model using every type of rating
# TODO: Big final tournaments to find out who wins at the end (+ RPP estimation)
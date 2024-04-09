import multiprocessing
from utils.policy import Policy
from Population import Population
from pettingzoo.classic import connect_four_v3

import datetime

print("imports ok")
# Hyperparameters

n_trials = 1
n_rounds = 10
n_opponents_per_agent = 40
num_fights_train = 1 # that's the number of fight against *each* opponent in training
num_fight_test = 0
n_agents = 15

# Rating systems we will use

rating_systems = ["elo", "bayeselo", "glicko", "glicko2", "trueskill", "melo"]

def train_population(rating_system):
    print(rating_system)
    # train DQN
    agent_counts = {
        Policy.DQN: n_agents,
        Policy.PPO: 0,
        Policy.A2C: 0,
        Policy.Random: 0,
        Policy.Deterministic:0
    }
    texas_population = Population(connect_four_v3.env(), agent_counts, num_trials=n_trials, num_rounds=n_rounds, num_opponnent_per_agent=n_opponents_per_agent, rating_system=rating_system, rating_systems=rating_systems)
    texas_population.training_loop(num_fights_train=num_fights_train, use_rating_in_reward=False)

if __name__ == '__main__':
    start = datetime.datetime.now()
    processes = []
    for rating_system in rating_systems:
        process = multiprocessing.Process(target=train_population, args=(rating_system,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print(f"Total experiment duration : {datetime.datetime.now() - start}")


    # train PPO

    # train A2C

# TODO: Track all ratings every time, we'll sort out the data once we have it
# TODO: Train every type of model using every type of rating
# TODO: Big final tournaments to find out who wins at the end (+ RPP estimation)
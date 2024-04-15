import json
import os
import numpy as np
import nashpy as nash
import warnings
import threading

# Define a function to run the Lemke-Howson enumeration with a timeout
def run_lemke_howson(game, result_holder, stop_event):
    try:
        result_holder['result'] = next(game.lemke_howson_enumeration())
    except StopIteration:
        result_holder['result'] = None
    except Exception as e:
        result_holder['error'] = e
    finally:
        stop_event.set()

# Load match data from the JSON file
with open("tournament_data.json", "r") as f:
    match_data = json.load(f)

# Extract player IDs and interactions
player_ids = match_data["ids"]
interactions = match_data["interactions"]

# Infer populations for each player based on file paths
player_populations = {}
for player_file, player_id in player_ids.items():
    if player_file.startswith("random"):
        # Handle agents with filepath "randomN"
        player_populations[player_id] = "random_agents"
    else:
        folder_name = os.path.dirname(player_file)
        player_populations[player_id] = folder_name

# Initialize payoff matrices for every pair of populations
payoff_matrices = {}

# Populate players for each population
populations = {}
for player_id, population_name in player_populations.items():
    if population_name not in populations:
        populations[population_name] = set()
    populations[population_name].add(player_id)

# Iterate through interactions to populate payoff matrices
for interaction in interactions:
    p0_id = interaction["p0"]
    p1_id = interaction["p1"]
    score_p0 = interaction["score0"]
    score_p1 = interaction["score1"]
    
    # Determine which populations each player belongs to
    p0_population = player_populations[p0_id]
    p1_population = player_populations[p1_id]
    
    # Create or update the payoff matrix for the pair of populations
    key = (p0_population, p1_population)
    if key not in payoff_matrices:
        payoff_matrices[key] = np.zeros((len(populations[p0_population]), len(populations[p1_population])))
        payoff_matrices[key[::-1]] = np.zeros((len(populations[p1_population]), len(populations[p0_population])))
    
    # Update the payoff matrix
    p0_index = list(populations[p0_population]).index(p0_id)
    p1_index = list(populations[p1_population]).index(p1_id)
    payoff_matrices[key][p0_index, p1_index] += score_p0
    payoff_matrices[key[::-1]][p1_index, p0_index] += score_p1

warnings.filterwarnings("ignore", category=RuntimeWarning)
rpps = {}

# Define timeout duration
timeout_duration = 5  # in seconds

for i, pop0 in enumerate(populations.keys()):
    for pop1 in list(populations.keys())[i+1:]:
        print(pop0, pop1)
        game = nash.Game(payoff_matrices[(pop0, pop1)], payoff_matrices[(pop1, pop0)].T)
        
        # Create a dictionary to hold the result
        result_holder = {}
        
        # Create an event to signal the thread to stop
        stop_event = threading.Event()
        
        # Run the Lemke-Howson enumeration with a timeout
        thread = threading.Thread(target=run_lemke_howson, args=(game, result_holder, stop_event))
        thread.start()
        
        # Wait for the timeout duration or until the thread finishes
        thread.join(timeout_duration)
        
        # If the thread is still running after the timeout, set the stop event
        if thread.is_alive():
            stop_event.set()
        
        if 'error' in result_holder:
            # If an error occurred during the execution, handle it
            print(f"Error occurred: {result_holder['error']}")
            rpps[(pop0, pop1)] = "error"
            rpps[(pop1, pop0)] = "error"
            continue
            
        eqs = result_holder.get('result')
        if eqs is None:
            rpps[(pop0, pop1)] = "TLE"
            rpps[(pop1, pop0)] = "TLE"
        
        try:
            rpp1 = eqs[0] @ payoff_matrices[(pop0, pop1)] @ eqs[1]
            rpp2 = eqs[1] @ payoff_matrices[(pop1, pop0)] @ eqs[0]
        except: 
            rpps[(pop0, pop1)] = "nan"
            rpps[(pop1, pop0)] = "nan"
        
        rpps[(pop0, pop1)] = rpp1
        rpps[(pop1, pop0)] = rpp2
        print(rpp1, rpp2)

sorted_populations = sorted(populations.keys())

# Create a dictionary to map population names to their indices
pop_index_map = {pop: index for index, pop in enumerate(sorted_populations)}

# Create a matrix to store RPPs
num_populations = len(pop_index_map)
rpp_matrix = np.zeros((num_populations, num_populations)).tolist()

# Populate the RPP matrix using the RPPs dictionary
for (pop0, pop1), rpp in rpps.items():
    pop0_index = pop_index_map[pop0]
    pop1_index = pop_index_map[pop1]
    rpp_matrix[pop0_index][pop1_index] = rpp

# Save the RPP matrix to a JSON file
rpp_data = {
    "rpp_matrix": rpp_matrix,
    "pop_index_map": pop_index_map
}

with open("rpp_data.json", "w") as json_file:
    json.dump(rpp_data, json_file)

print("RPP matrix and population index map saved to rpp_data.json")
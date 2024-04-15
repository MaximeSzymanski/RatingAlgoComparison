import json
import os
import numpy as np

with open("tournament_ratings.json", "r") as f:
    data = json.load(f)


# Initialize dictionaries to store populations and statistics
populations = {}
population_stats = {}

# Iterate through the dictionary keys
for agent_file, agent_data in data.items():
    # Extract the folder name from the file path
    folder_name = os.path.dirname(agent_file)
    
    # Check if the folder name corresponds to "randomN" format
    if agent_file.startswith("random"):
        # Handle agents with filepath "randomN"
        population_name = "random_agents"
    else:
        # Handle agents with regular filepaths
        population_name = folder_name

    # Update populations dictionary
    if population_name in populations:
        populations[population_name].append(agent_file)
    else:
        populations[population_name] = [agent_file]

    # Compute statistics for each population
    if population_name not in population_stats:
        population_stats[population_name] = {rating_type: [] for rating_type in agent_data.keys()}
    
    for rating_type, rating_value in agent_data.items():
        population_stats[population_name][rating_type].append(rating_value)

# Compute mean and standard deviation for each rating type in each population
for population_name, stats in population_stats.items():
    for rating_type, rating_values in stats.items():
        population_stats[population_name][rating_type] = {
            'mean': np.mean(rating_values),
            'std_dev': np.std(rating_values)
        }

mean_bayeselo_ratings = {population_name: stats["bayeselo"]["mean"] 
                         for population_name, stats in population_stats.items()}

sorted_populations = sorted(mean_bayeselo_ratings.items(), key=lambda x: x[1], reverse=True)

# Print markdown table header
print("| Population | Mean Elo (± Std Dev) | Mean BayesElo (± Std Dev) | Mean Glicko (± Std Dev) | Mean Glicko2 (± Std Dev) | Mean Trueskill (± Std Dev) | Mean Melo2 (± Std Dev) | Mean Winrate (± Std Dev) | Mean Drawrate (± Std Dev) | Mean Loserate (± Std Dev) | Mean WDL (± Std Dev) | Mean Wins (± Std Dev) | Mean Draws (± Std Dev) | Mean Losses (± Std Dev) |")
print("|------------|-----------------------|----------------------------|-------------------------|--------------------------|-----------------------------|-------------------------|---------------------------|----------------------------|--------------------------|-----------------------|-----------------------|-----------------------|")

# Iterate through sorted populations and print table rows
for population_name, mean_bayeselo_rating in sorted_populations:
    metrics_str = ""
    for metric_name in ["elo", "bayeselo", "glicko", "glicko2", "trueskill", "melo2", "winrate", "drawrate", "loserate", "wdl", "wins", "draws", "losses"]:
        metric_stats = population_stats[population_name][metric_name]
        metric_str = f"{metric_stats['mean']:.2f} ± {metric_stats['std_dev']:.2f}"
        metrics_str += f" | {metric_str}"
    print(f"| {population_name} {metrics_str} |")

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load RPP data from JSON file
with open("rpp_data.json", "r") as json_file:
    rpp_data = json.load(json_file)

# Extract RPP matrix and population index map
rpp_matrix = rpp_data["rpp_matrix"]
pop_index_map = rpp_data["pop_index_map"]
populations = list(pop_index_map.keys())

# Convert invalid values to NaN
for i in range(len(rpp_matrix)):
    for j in range(len(rpp_matrix[i])):
        if rpp_matrix[i][j] == 'error':
            rpp_matrix[i][j] = np.nan

# Convert RPP matrix to np.float64
rpp_matrix = np.array(rpp_matrix, dtype=np.float64)

# Reorder the RPP matrix to match populations sorted by mean BayesElo rating
sorted_populations = sorted(populations, key=lambda x: mean_bayeselo_ratings.get(x, 0), reverse=True)
sorted_indices = [pop_index_map[pop] for pop in sorted_populations]
sorted_rpp_matrix = rpp_matrix[sorted_indices][:, sorted_indices]

# Create a color matrix for plotting
color_matrix = np.zeros_like(sorted_rpp_matrix, dtype=np.float64)

# Normalize RPP values to range [0, 1]
rpp_min = np.nanmin(sorted_rpp_matrix)
rpp_max = np.nanmax(sorted_rpp_matrix)
normalized_rpp_matrix = (sorted_rpp_matrix - rpp_min) / (rpp_max - rpp_min)

# Set colors based on normalized RPP values, and grey for NaN values
for i in range(len(populations)):
    for j in range(len(populations)):
        if np.isnan(normalized_rpp_matrix[i][j]):
            color_matrix[i][j] = np.nan
        else:
            color_matrix[i][j] = normalized_rpp_matrix[i][j]

# Create reversed colormap (red for low values, green for high values)
cmap = plt.cm.get_cmap('coolwarm_r')
norm = mcolors.Normalize(vmin=0, vmax=1)

# Compute row sums
row_sums = np.nansum(color_matrix, axis=1)

# Convert labels
labels = [s.replace("saved_models\\", "").replace("using_", "").replace("_for_matchmaking", "") for s in sorted_populations]

# Plot the colored matrix
plt.figure(figsize=(10, 8))
plt.imshow(color_matrix, cmap=cmap, norm=norm, interpolation='nearest')
cbar = plt.colorbar(label='Relative Population Performance')
cbar.set_ticks(np.linspace(0, 1, 11))  # Set ticks for color bar
cbar.set_ticklabels(['0', '', '', '', '', '', '', '', '', '', '1'])  # Set tick labels
plt.xticks(ticks=np.arange(len(populations)), labels=labels, rotation=90)
plt.yticks(ticks=np.arange(len(populations)), labels=labels)  # Reverse the order
plt.title('Relative Population Performance Matrix')
plt.tight_layout()

# Save the plot
plt.savefig('rpp_matrix.png', dpi=300)

# Print markdown table
print("| Population | Sum of RPP |")
print("|------------|------------|")
for label, row_sum in zip(labels, row_sums):
    print(f"| {label} | {row_sum:.2f} |")

# Show the plot
plt.show()

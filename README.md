
# 🚀 The Impact of rating systems on Multi-Agent Reinforcement Learning

## 📝 Description
This project contains several algorithms and models implemented in Python for comparison and analysis. The main focus is on rating algorithms and their performance in different scenarios.

## 📂 Project Structure
The project is organized into several Python files, each serving a specific purpose:

- `Agent.py`: This file contains the `Agent` class which is used to represent an agent in the environment.
- `models/`: This directory contains various models including `PPO`, `DQN`, `A2C`, `Random`, and `Deterministic`.
- `rating/`: This directory contains the `RatingSystem` class along with its subclasses `Elo`, `Glicko`,`Glicko2`, and `TrueSkill`.
- `utils/`: This directory contains various utility functions and classes for plotting, logging, and policy handling.
- `utils/plot.py`: This file contains functions for plotting various data such as win rates, strategy landscapes, and diversity matrices.
- `utils/matchmaking.py`: This file contains the `Prioritized_fictitious_plays` class which is used for matchmaking in the environment.
- `utils/diversity_action.py`: This file contains the `Diversity` class which is used to calculate the diversity of the population of agents.
- `utils/logger.py`: This file contains the `Logger` class which is used for logging various data and results.

## 🛠️ Installation
To install the necessary dependencies, run the following command in your terminal:
```bash
pip install -r requirements.txt
```

## 🎯 Usage
To run the project, execute the following command in your terminal:
```bash
python run_experiment.py
```


## 📜 License
This project is licensed under the [MIT License](LICENSE).

## 📫 Contact
Maxime Szymanski - maxime.szymanski@mail.mcgill.ca
Aurélien Bück-Kaeffer - 

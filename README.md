# RatingAlgoComparison

In order to run this code, first you need to `pip install -r requirements.txt`. In case poprank was not installed properly by doing this, you need to git clone `https://github.com/poprl/poprank.git`, cd to the folder where it was cloned and type `pip install -e .` (don't forget the dot at the end of the command). 

Running `run_experiments.py` will train all populations of agents. You may change the hyperparameters in the file. This may take very long (in the ballpark of 12h). This uses multiprocessing and cuda so it should more or less scale with the compute you have available. Once training is complete, the folders `logs` and `saved_models` will contain the results of the training.

After that, you can run `tournament.py` which will make the agents trained in the last step play a tournament and log the raw results in `tournament_data.json`. this may also take very long (in the ballpark of 45min).

Then, running `tournament_rating_computation.py` will compute the ratings of each player and save the results in `tournaùent_ratings.json`.

Finally, tunning `tournament_results_analysis.py` will perform some data analysis on the results like computing mean and std ratings for each population, compute a RPP matrix...
import random
from trueskill import Rating, rate_1vs1
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import itertools
import trueskill


class RatingSystem:
    def __init__(self):
        self.ratings = {}
        self.base_rating = 0

    def add_player(self, player_id: int):
        raise NotImplementedError

    def remove_player(self, player_id: int):
        """
        Remove a player from the Elo rating system.

        Parameters:
        - player_id (int): The unique identifier for the player.
        """
        if player_id in self.ratings:
            del self.ratings[player_id]

    def get_rating(self, player_id: int, to_plot=False) -> int:
        raise NotImplementedError

    def plot_rating_per_policy(self, policies: List[str], rating_mean: Dict[str, List[int]],
                               rating_std: Dict[str, List[int]]) -> None:
        """
        Plot the  rating of each policy over time.

        Parameters:
        - policies (List[str]): A list of policy names.
        - elos_mean (Dict[str, List[int]]): A dictionary containing the mean Elo ratings for each policy.
        - elos_std (Dict[str, List[int]]): A dictionary containing the standard deviations of Elo ratings for each policy.
        """
        raise NotImplementedError

    def update_ratings(self, winner_id: int, loser_id: int):
        """
        Update the ratings of two players after a match.

        Parameters:
        - winner_id (int): The unique identifier for the winning player.
        - loser_id (int): The unique identifier for the losing player.
        """
        raise NotImplementedError


class Elo(RatingSystem):
    def __init__(self, k_factor: int = 32):
        """
        Initialize the Elo rating system.

        Parameters:
        - k_factor (int): The K-factor, representing the sensitivity of the Elo rating system.
          Default value is 32.
        """
        super().__init__()
        self.k_factor = k_factor
        self.base_rating = 1000

    def plot_rating_per_policy(self, policies: List[str], rating_mean: Dict[str, List[int]],
                               rating_std: Dict[str, List[int]]) -> None:
        """
        Plot the Elo rating of each policy over time.

        Parameters:
        - policies (List[str]): A list of policy names.
        - elos_mean (Dict[str, List[int]]): A dictionary containing the mean Elo ratings for each policy.
        - elos_std (Dict[str, List[int]]): A dictionary containing the standard deviations of Elo ratings for each policy.
        """

        for key in rating_mean.keys():
            rating_mean[key] = np.mean(np.array(rating_mean[key]), axis=1)

        for key in rating_std.keys():
            rating_std[key] = np.std(np.array(rating_std[key]), axis=1)

        # Define a larger set of distinct colors using a colormap
        colors = plt.cm.get_cmap('tab20', len(policies))
        for idx, policy in enumerate(policies):
            # Remove "Policy." from the policy name
            policy_name = str(policy)

            plt.plot(range(len(
                rating_mean[policy])), rating_mean[policy], label=policy_name, color=colors(idx))

            x = np.array(range(len(rating_mean[policy])))
            y_mean = np.array(rating_mean[policy])
            y_std = np.array(rating_std[policy])
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3)
        # increase the size of the plot
        plt.gcf().set_size_inches(20, 15)

        plt.xlabel("Number of Fights")
        plt.ylabel("Rating")
        plt.legend()
        # put legend top left
        plt.legend(loc='upper left')
        plt.title("Rating Over Time")
        plt.savefig('rating.png')
        plt.clf()

    def update_ratings(self, player1_id: int, player2_id: int, draw=False):
        """
        Update the ratings of two players after a match.

        Parameters:
        - player1_id (int): The unique identifier for one of the players.
        - player2_id (int): The unique identifier for the other player.
        - draw (bool, optional): Whether the match ended in a draw. Defaults to False.
        """

        player1_rating = self.get_rating(player1_id)
        player2_rating = self.get_rating(player2_id)

        if draw:
            # compute the probability of a draw
            expected_draw = self._expected_result(
                player1_rating, player2_rating)

            # update the ratings
            player1_new_rating = player1_rating + \
                self.k_factor * (0.5 - expected_draw)
            player2_new_rating = player2_rating + \
                self.k_factor * (0.5 - expected_draw)
        else:
            expected_win = self._expected_result(
                player1_rating, player2_rating)
            player1_new_rating = player1_rating + \
                self.k_factor * (1 - expected_win)
            player2_new_rating = player2_rating + \
                self.k_factor * (0 - (1 - expected_win))

        self.ratings[player1_id] = player1_new_rating
        self.ratings[player2_id] = player2_new_rating

    def get_rating(self, player_id: int, to_plot=False) -> int:
        """
        Get the current Elo rating of a player.

        Parameters:
        - player_id (int): The unique identifier for the player.

        Returns:
        - int: The Elo rating of the player.
        """
        return self.ratings.get(player_id, self.base_rating)

    def _expected_result(self, player_rating: int, opponent_rating: int) -> float:
        """
        Calculate the expected outcome of a match.

        Parameters:
        - player_rating (int): The rating of the player.
        - opponent_rating (int): The rating of the opponent.

        Returns:
        - float: The expected probability of the player winning.
        """
        return 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))

    def find_similar_rating_pairs(self) -> list:
        """
        Find pairs of agents with similar Elo ratings, each having one opponent.

        Returns:
        - list of tuples: A list of tuples where each tuple represents a pair of agents,
                          with each agent having one opponent.
        """
        players = list(self.ratings.keys())
        random.shuffle(players)  # Shuffle players to randomize pairing

        paired_agents = []
        used_players = set()

        for player in players:
            if player not in used_players:
                opponent = self._find_opponent(player, used_players)
                if opponent:
                    paired_agents.append((player, opponent))
                    used_players.add(player)
                    used_players.add(opponent)

        return paired_agents

    def add_player(self, player_id: int):
        """
        Add a player to the Elo rating system with an initial rating.

        Parameters:
        - player_id (int): The unique identifier for the player.
        """
        if player_id not in self.ratings:
            self.ratings[player_id] = self.base_rating

    def _find_opponent(self, player: int, used_players: set) -> int:
        """
        Find an opponent for a player with a similar Elo rating.

        Parameters:
        - player (int): The unique identifier for the player.
        - used_players (set): A set containing the identifiers of players already paired.

        Returns:
        - int: The identifier of the opponent for the given player.
        """
        player_rating = self.get_rating(player)
        remaining_players = set(self.ratings.keys()) - used_players
        opponents = []

        for opponent in remaining_players:
            if opponent != player:
                opponent_rating = self.get_rating(opponent)
                if abs(player_rating - opponent_rating) <= 100:  # Adjust threshold as needed
                    opponents.append(opponent)

        if opponents:
            return random.choice(opponents)
        else:
            return None  # No suitable opponent found

    def plot_elo_distribution(self, round=0):
        """
        Plot the distribution of Elo ratings for all players.
        """
        import matplotlib.pyplot as plt

        ratings = list(self.ratings.values())
        plt.hist(ratings, bins=20, edgecolor='black')
        plt.xlabel('Elo Rating')
        plt.ylabel('Frequency')
        plt.title('Elo Rating Distribution')
        # save into file
        plt.savefig('elo_distrib/elo_distribution_round_'+str(round)+'.png')
        plt.clf()


class TrueSkill(RatingSystem):
    def __init__(self):
        super().__init__()

    def add_player(self, player_id: int):
        """
        Add a player to the TrueSkill rating system with an initial rating.

        Parameters:
        - player_id (int): The unique identifier for the player.
        - rating (Rating): The initial rating for the player. Default is a new Rating instance.
        """

        if player_id not in self.ratings:
            self.ratings[player_id] = Rating()

    def find_similar_rating_pairs(self) -> list:
        """
        Find pairs of agents with similar TrueSkill ratings, each having one opponent.

        Returns:
        - list of tuples: A list of tuples where each tuple represents a pair of agents,
                          with each agent having one opponent.
        """
        # Get all combinations of players
        all_combinations = list(itertools.combinations(self.ratings.keys(), 2))

        # Calculate the difference in ratings for each combination
        rating_differences = []
        for player1, player2 in all_combinations:
            rating_difference = abs(
                trueskill.expose(self.get_trueskill_ratings(player1)) - trueskill.expose(self.get_trueskill_ratings(player2)))
            rating_differences.append((player1, player2, rating_difference))

        # Sort combinations by rating difference
        rating_differences.sort(key=lambda x: x[2])

        # Pair up agents with similar ratings
        paired_agents = []
        used_players = set()
        for player1, player2, _ in rating_differences:
            if player1 not in used_players and player2 not in used_players:
                paired_agents.append((player1, player2))
                used_players.add(player1)
                used_players.add(player2)

        return paired_agents

    def update_ratings(self, winner_id: int, loser_id: int, draw=False):
        """
        Update the ratings of two players after a match.

        Parameters:
        - winner_id (int): The unique identifier for the winning player.
        - loser_id (int): The unique identifier for the losing player.
        """
        winner_rating = self.get_trueskill_ratings(winner_id)
        loser_rating = self.get_trueskill_ratings(loser_id)

        if draw:
            winner, loser = rate_1vs1(winner_rating, loser_rating, drawn=True)
        else:
            winner, loser = rate_1vs1(winner_rating, loser_rating)

        self.ratings[winner_id] = winner
        self.ratings[loser_id] = loser

    def get_trueskill_ratings(self, player_id):
        """
        Get the current TrueSkill ratings of a player.

        Returns:
        - dict: A dictionary containing the TrueSkill ratings of all players.
        """
        return self.ratings.get(player_id)

    def get_rating(self, player_id: int, to_plot=False) -> float:
        """
        Get the current TrueSkill rating of a player.

        Parameters:
        - player_id (int): The unique identifier for the player.

        Returns:
        - Rating: The TrueSkill rating of the player.
        """
        if to_plot:
            return self.get_trueskill_ratings(player_id)
        else:
            return self.ratings.get(player_id).mu - 3 * self.ratings.get(player_id).sigma

    def plot_rating_per_policy(self, policies: List[str], rating_mean: Dict[str, List[Rating]],
                               rating_std: Dict[str, List[Rating]]) -> None:
        """
        Plot the TrueSkill rating of each policy over time.

        Parameters:
        - policies (List[str]): A list of policy names.
        - rating_mean (Dict[str, List[int]]): A dictionary containing the mean TrueSkill ratings for each policy.
        - rating_std (Dict[str, List[int]]): A dictionary containing the standard deviations of TrueSkill ratings for each policy.
        """
        for key in rating_mean.keys():
            rating_mean[key] = np.mean(np.array(rating_mean[key]), axis=1)

        for key in rating_std.keys():
            rating_std[key] = np.std(np.array(rating_std[key]), axis=1)

        colors = plt.cm.get_cmap('tab20', len(policies))
        for idx, policy in enumerate(policies):
            # Remove "Policy." from the policy name
            policy_name = str(policy)

            plt.plot(range(len(
                rating_mean[policy])), rating_mean[policy], label=policy_name, color=colors(idx))

            x = np.array(range(len(rating_mean[policy])))
            y_mean = np.array(rating_mean[policy])
            y_std = np.array(rating_std[policy])
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3)
        # increase the size of the plot
        plt.gcf().set_size_inches(20, 15)

        plt.xlabel("Number of Fights")
        plt.ylabel("Rating")
        plt.legend()
        # put legend top left
        plt.legend(loc='upper left')
        plt.title("Rating Over Time")
        plt.savefig('rating.png')
        plt.clf()

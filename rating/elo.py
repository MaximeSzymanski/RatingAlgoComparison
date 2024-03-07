import random

class Elo:
    def __init__(self, k_factor: int = 32):
        """
        Initialize the Elo rating system.

        Parameters:
        - k_factor (int): The K-factor, representing the sensitivity of the Elo rating system.
          Default value is 32.
        """
        self.k_factor = k_factor
        self.ratings = {}

    def add_player(self, player_id: int, rating: int = 1500):
        """
        Add a player to the Elo rating system with an initial rating.

        Parameters:
        - player_id (int): The unique identifier for the player.
        - rating (int): The initial rating for the player. Default is 1500.
        """
        if player_id not in self.ratings:
            self.ratings[player_id] = rating

    def get_rating(self, player_id: int) -> int:
        """
        Get the current rating of a player.

        Parameters:
        - player_id (int): The unique identifier for the player.

        Returns:
        - int: The current rating of the player.
        """
        return self.ratings.get(player_id, 1500)

    def update_ratings(self, winner_id: int, loser_id: int):
        """
        Update the ratings of two players after a match.

        Parameters:
        - winner_id (int): The unique identifier for the winning player.
        - loser_id (int): The unique identifier for the losing player.
        """
        winner_rating = self.get_rating(winner_id)
        loser_rating = self.get_rating(loser_id)

        expected_win = self._expected_result(winner_rating, loser_rating)
        winner_new_rating = winner_rating + self.k_factor * (1 - expected_win)
        loser_new_rating = loser_rating + self.k_factor * (0 - (1 - expected_win))

        self.ratings[winner_id] = winner_new_rating
        self.ratings[loser_id] = loser_new_rating

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

    def find_similar_elo_pairs(self) -> list:
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

    def plot_elo_distribution(self,round=0):
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
        plt.savefig('elo_distribution_round_'+str(round)+'.png')
        plt.clf()
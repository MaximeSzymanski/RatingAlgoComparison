import json
from popcore import Interaction
import random

from poprank.functional.rates.elo import EloRate, elo
from poprank.functional.rates.glicko import GlickoRate, Glicko2Rate, glicko, glicko2
from poprank.functional.rates.trueskill import TrueSkillRate, trueskill
from poprank.functional.rates.melo import MultidimEloRate, multidim_elo
from poprank.functional.rates.bayeselo import bayeselo
from poprank.functional.rates.wdl import windrawlose, winlose
from poprank import Rate

with open("tournament_data.json", "r") as f:
    data = json.load(f)

random.seed(4)
player_to_id = data["ids"]
id_to_player = [0 for i in player_to_id]
for player, id in player_to_id.items():
    id_to_player[id] = player
players = list(player_to_id.keys())

matches = data["interactions"]
random.shuffle(matches)
interactions = [Interaction([id_to_player[i["p0"]], id_to_player[i["p1"]]], [i["score0"], i["score1"]]) for i in matches]

# Initialize the elos to 0
players = list(players)
elos = [EloRate(0) for x in players]
bayeselos = [EloRate(0) for x in players]
glickos = [GlickoRate(0) for x in players]
glickos2 = [Glicko2Rate(0) for x in players]
trueskills = [TrueSkillRate(25, 25/3) for x in players]
wdl = [Rate(0) for x in players]
wins = [Rate(0) for x in players]
draws = [Rate(0) for x in players]
losses = [Rate(0) for x in players]
played = [Rate(0) for x in players]
melos2 = [MultidimEloRate(0, 1, k=1) for x in players]

# Compute the ratings
print("elo")
elos = elo(players, interactions, elos, k_factor=4)
print("bayeselo")
bayeselos = bayeselo(players, interactions, bayeselos)
print("glicko")
glickos = glicko(players, interactions, glickos)
#glickos2 = glicko2(players, interactions, glickos2)
print("trueskill")
trueskills = trueskill(players, interactions, trueskills)
print("windrawlose")
wdl = windrawlose(players, interactions, wins, 1, 0, -1)
wins = windrawlose(players, interactions, wins, 1, 0, 0)
draws = windrawlose(players, interactions, draws, 0, 1, 0)
losses = windrawlose(players, interactions, losses, 0, 0, 1)
played = windrawlose(players, interactions, played, 1, 1, 1)
print("melos2")
melos2 = multidim_elo(players, interactions, melos2, k=1, lr1=0.0001, lr2=0.01)

print("sequential elo")
sequential_elo = [EloRate(0) for x in players]
for match in interactions:
    sequential_elo = elo(players, [match], sequential_elo, k_factor=4)

# Rank the players based on their bayeselo ratings
players, elos, bayeselos, selo, glickos, glickos2, trueskills, wdl, wins, \
    draws, losses, played, melos2 = \
    [list(t) for t in zip(*sorted(
        zip(players, elos, bayeselos, sequential_elo, glickos, glickos2,
            trueskills, wdl, wins, draws, losses, played, melos2),
        key=lambda x: x[8].mu/x[11].mu if x[11].mu else 0, reverse=True))]


json_data = {}
# Print the results
for p, e, b, se, g, g2, t, wdl_, w, d, l, pl, ml2 in\
   zip(players, elos, bayeselos, selo, glickos, glickos2, trueskills, wdl,
       wins, draws, losses, played, melos2):
    print(
        f"| {p} | {round(b.mu, 1)} | {round(se.mu, 1)} | "
        f"{round(g.mu, 1)} | {round(g2.mu, 1)} | {round(t.mu, 1)} | "
        f"{round(ml2.mu, 5)} | {round(w.mu/pl.mu*100 if pl.mu else 0, 1)} |"
        f"{round(d.mu/pl.mu*100 if pl.mu else 0, 1)} | "
        f"{round(l.mu/pl.mu*100 if pl.mu else 0, 1)} | {wdl_.mu} | {w.mu} | "
        f"{d.mu} | {l.mu} | {pl.mu}")

    json_data[p] = {"elo" : se.mu, "bayeselo": b.mu, "glicko" : g.mu, "glicko2": g2.mu, "trueskill" : t.mu, \
        "melo2" : float(ml2.mu), "winrate": w.mu/pl.mu*100 if pl.mu else 0, "drawrate": d.mu/pl.mu*100 if pl.mu else 0, \
        "loserate": l.mu/pl.mu*100 if pl.mu else 0, "wdl" : wdl_.mu, "wins": w.mu, "draws": d.mu, "losses": l.mu, "played":pl.mu}

with open("tournament_ratings.json", "w") as f:
    f.write(json.dumps(json_data))
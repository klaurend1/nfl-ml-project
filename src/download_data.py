import nflreadpy as nfl

# Load every season we need
seasons = list(range(2000, 2027))

games = nfl.load_schedules(seasons)

print(games.shape)
print(games.columns)
print(games.head())

games.write_csv("nfl_games_2000_2026.csv")
print("\nSaved nfl_games_2000_2026.csv")
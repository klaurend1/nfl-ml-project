import pandas as pd


# --------------------------------------------------
# FILES
# --------------------------------------------------

GAMES_FILE = "nfl_games_2000_2026.csv"
PRESEASON_FILE = "nfl_preseason_2000_2026.csv"


# --------------------------------------------------
# NORMALIZE HISTORICAL TEAM CODES
# --------------------------------------------------

TEAM_MAP = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "LA": "LAR",
    "WSH": "WAS",
    "JAC": "JAX"
}


def normalize_team(team):
    return TEAM_MAP.get(team, team)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

games = pd.read_csv(GAMES_FILE)
preseason = pd.read_csv(PRESEASON_FILE)

games["away_team"] = games["away_team"].apply(normalize_team)
games["home_team"] = games["home_team"].apply(normalize_team)


# --------------------------------------------------
# HISTORICAL SEASONS ONLY
# --------------------------------------------------
# 2020 is excluded because there was no preseason.
# 2026 is excluded because we want it to remain unseen data.

games = games[
    (games["season"] >= 2000)
    & (games["season"] <= 2025)
    & (games["season"] != 2020)
].copy()


# --------------------------------------------------
# REGULAR-SEASON GAMES
# --------------------------------------------------

regular = games[
    games["game_type"] == "REG"
].copy()

regular = regular.dropna(
    subset=["away_score", "home_score"]
)


# --------------------------------------------------
# BUILD TEAM-LEVEL REGULAR-SEASON RECORDS
# --------------------------------------------------

records = {}

for _, game in regular.iterrows():

    season = int(game["season"])

    away = game["away_team"]
    home = game["home_team"]

    away_score = int(game["away_score"])
    home_score = int(game["home_score"])

    for team in [away, home]:

        key = (season, team)

        if key not in records:
            records[key] = {
                "season": season,
                "team": team,
                "reg_wins": 0,
                "reg_losses": 0,
                "reg_ties": 0
            }

    if away_score > home_score:

        records[(season, away)]["reg_wins"] += 1
        records[(season, home)]["reg_losses"] += 1

    elif home_score > away_score:

        records[(season, home)]["reg_wins"] += 1
        records[(season, away)]["reg_losses"] += 1

    else:

        records[(season, away)]["reg_ties"] += 1
        records[(season, home)]["reg_ties"] += 1


# --------------------------------------------------
# CONVERT TO DATAFRAME
# --------------------------------------------------

outcomes = pd.DataFrame(records.values())


# --------------------------------------------------
# REGULAR-SEASON METRICS
# --------------------------------------------------

outcomes["reg_games"] = (
    outcomes["reg_wins"]
    + outcomes["reg_losses"]
    + outcomes["reg_ties"]
)

outcomes["reg_win_pct"] = (
    outcomes["reg_wins"]
    + 0.5 * outcomes["reg_ties"]
) / outcomes["reg_games"]

outcomes["reg_win_pct"] = (
    outcomes["reg_win_pct"].round(4)
)


# --------------------------------------------------
# PLAYOFF TEAMS
# --------------------------------------------------

playoff_types = [
    "WC",
    "DIV",
    "CON",
    "SB"
]

playoff_games = games[
    games["game_type"].isin(playoff_types)
].copy()

playoff_teams = set()

for _, game in playoff_games.iterrows():

    season = int(game["season"])

    playoff_teams.add(
        (season, game["away_team"])
    )

    playoff_teams.add(
        (season, game["home_team"])
    )


outcomes["made_playoffs"] = outcomes.apply(
    lambda row: int(
        (row["season"], row["team"])
        in playoff_teams
    ),
    axis=1
)


# --------------------------------------------------
# SUPER BOWL CHAMPIONS
# --------------------------------------------------

super_bowls = games[
    games["game_type"] == "SB"
].copy()

champions = set()

for _, game in super_bowls.iterrows():

    season = int(game["season"])

    away = game["away_team"]
    home = game["home_team"]

    away_score = int(game["away_score"])
    home_score = int(game["home_score"])

    if away_score > home_score:
        winner = away
    else:
        winner = home

    champions.add(
        (season, winner)
    )


outcomes["super_bowl_champion"] = outcomes.apply(
    lambda row: int(
        (row["season"], row["team"])
        in champions
    ),
    axis=1
)


# --------------------------------------------------
# SORT OUTCOME DATA
# --------------------------------------------------

outcomes = outcomes.sort_values(
    ["season", "team"]
).reset_index(drop=True)


# --------------------------------------------------
# SAVE REGULAR-SEASON OUTCOMES
# --------------------------------------------------

outcomes.to_csv(
    "nfl_regular_season_outcomes.csv",
    index=False
)


# --------------------------------------------------
# MERGE WITH PRESEASON FEATURES
# --------------------------------------------------

final = preseason.merge(
    outcomes,
    on=["season", "team"],
    how="left"
)

final = final.sort_values(
    ["season", "team"]
).reset_index(drop=True)


# --------------------------------------------------
# SAVE FINAL ML DATASET
# --------------------------------------------------

final.to_csv(
    "nfl_ml_dataset.csv",
    index=False
)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET BUILD COMPLETE")
print("=" * 60)

print(
    "\nRegular-season outcomes shape:",
    outcomes.shape
)

print(
    "Final dataset shape:",
    final.shape
)


# --------------------------------------------------
# DUPLICATES
# --------------------------------------------------

duplicates = final.duplicated(
    subset=["season", "team"]
).sum()

print(
    "\nDuplicate team-seasons:",
    duplicates
)


# --------------------------------------------------
# SUPER BOWL CHAMPION CHECK
# --------------------------------------------------

champion_counts = (
    outcomes.groupby("season")
    ["super_bowl_champion"]
    .sum()
)

print(
    "\nSuper Bowl champions per season:"
)

print(champion_counts)


# --------------------------------------------------
# TEAM COUNTS PER SEASON
# --------------------------------------------------

team_counts = (
    outcomes.groupby("season")
    .size()
)

print(
    "\nRegular-season teams per season:"
)

print(team_counts)


# --------------------------------------------------
# HISTORICAL MISSING TARGETS
# --------------------------------------------------

historical = final[
    final["season"] <= 2025
]

missing_targets = historical[
    "reg_wins"
].isna().sum()

print(
    "\nHistorical rows missing regular-season wins:",
    missing_targets
)


# --------------------------------------------------
# SHOW EXACTLY WHICH HISTORICAL ROWS ARE MISSING
# --------------------------------------------------

missing_rows = final[
    (final["season"] <= 2025)
    & (final["reg_wins"].isna())
][["season", "team"]]

print(
    "\nMissing historical target rows:"
)

if len(missing_rows) == 0:
    print("None")
else:
    print(
        missing_rows.to_string(
            index=False
        )
    )


# --------------------------------------------------
# CHECK 2026 TARGETS
# --------------------------------------------------

future = final[
    final["season"] == 2026
]

print(
    "\n2026 rows:",
    len(future)
)

print(
    "2026 rows with regular-season targets:",
    future["reg_wins"].notna().sum()
)


# --------------------------------------------------
# 2000 BALTIMORE SANITY CHECK
# --------------------------------------------------

print(
    "\n2000 Baltimore:"
)

print(
    final[
        (final["season"] == 2000)
        & (final["team"] == "BAL")
    ].to_string(index=False)
)


# --------------------------------------------------
# 2026 SAMPLE
# --------------------------------------------------

print(
    "\nFirst five 2026 teams:"
)

print(
    final[
        final["season"] == 2026
    ].head().to_string(index=False)
)


# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print(
    "\nSaved:"
    "\n  nfl_regular_season_outcomes.csv"
    "\n  nfl_ml_dataset.csv"
)
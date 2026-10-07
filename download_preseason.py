import requests
import pandas as pd
import time


# --------------------------------------------------
# ESPN TEAM IDs
# --------------------------------------------------
# We use ESPN's numeric team IDs instead of abbreviations.
# This avoids problems with historical team names such as:
# OAK -> LV
# SD  -> LAC
# STL -> LAR
# WSH -> WAS

teams = {
    "ARI": 22,
    "ATL": 1,
    "BAL": 33,
    "BUF": 2,
    "CAR": 29,
    "CHI": 3,
    "CIN": 4,
    "CLE": 5,
    "DAL": 6,
    "DEN": 7,
    "DET": 8,
    "GB": 9,
    "HOU": 34,
    "IND": 11,
    "JAX": 30,
    "KC": 12,
    "LV": 13,
    "LAC": 24,
    "LAR": 14,
    "MIA": 15,
    "MIN": 16,
    "NE": 17,
    "NO": 18,
    "NYG": 19,
    "NYJ": 20,
    "PHI": 21,
    "PIT": 23,
    "SEA": 26,
    "SF": 25,
    "TB": 27,
    "TEN": 10,
    "WAS": 28
}


# --------------------------------------------------
# SEASONS
# --------------------------------------------------
# 2020 is excluded because the NFL had no preseason.

seasons = [
    year for year in range(2000, 2027)
    if year != 2020
]


# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def get_score(competitor):
    """
    Safely extract a team's score from ESPN's JSON.
    """

    score = competitor.get("score")

    if score is None:
        return None

    if isinstance(score, dict):
        value = score.get("value")

        if value is None:
            value = score.get("displayValue")

        if value is None:
            return None

        return int(float(value))

    return int(float(score))


# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------

rows = []
errors = []

for year in seasons:

    print(f"\nDownloading {year} preseason...")

    for team, espn_id in teams.items():

        url = (
            "https://site.api.espn.com/apis/site/v2/"
            f"sports/football/nfl/teams/{espn_id}/schedule"
            f"?season={year}&seasontype=1"
        )

        try:

            response = requests.get(
                url,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            events = data.get("events", [])

            # Some franchises did not exist yet.
            # Example: Houston Texans before 2002.
            if not events:
                continue


            # ------------------------------------------
            # TEAM-SEASON TOTALS
            # ------------------------------------------

            wins = 0
            losses = 0
            ties = 0

            points_for = 0
            points_against = 0

            games_played = 0


            # ------------------------------------------
            # PROCESS EACH PRESEASON GAME
            # ------------------------------------------

            for event in events:

                competition = event["competitions"][0]

                # Ignore cancelled or incomplete games.
                status = (
                    competition
                    .get("status", {})
                    .get("type", {})
                )

                completed = status.get("completed", False)

                if not completed:
                    continue


                competitors = competition.get(
                    "competitors",
                    []
                )

                team_data = None
                opponent_data = None


                # --------------------------------------
                # IDENTIFY OUR TEAM USING ESPN ID
                # --------------------------------------

                for competitor in competitors:

                    competitor_id = int(
                        competitor["team"]["id"]
                    )

                    if competitor_id == espn_id:
                        team_data = competitor
                    else:
                        opponent_data = competitor


                if team_data is None:
                    continue

                if opponent_data is None:
                    continue


                # --------------------------------------
                # SCORES
                # --------------------------------------

                team_score = get_score(team_data)
                opponent_score = get_score(opponent_data)

                if team_score is None:
                    continue

                if opponent_score is None:
                    continue


                # --------------------------------------
                # UPDATE TOTALS
                # --------------------------------------

                games_played += 1

                points_for += team_score
                points_against += opponent_score


                if team_score > opponent_score:
                    wins += 1

                elif team_score < opponent_score:
                    losses += 1

                else:
                    ties += 1


            # No completed preseason games
            if games_played == 0:
                continue


            # ------------------------------------------
            # DERIVED FEATURES
            # ------------------------------------------

            win_pct = (
                wins + 0.5 * ties
            ) / games_played

            ppg = (
                points_for
                / games_played
            )

            pa_pg = (
                points_against
                / games_played
            )

            point_diff_pg = (
                points_for - points_against
            ) / games_played


            # ------------------------------------------
            # SAVE TEAM-SEASON ROW
            # ------------------------------------------

            rows.append({

                "season": year,

                # Our normalized team code
                "team": team,

                "pre_games": games_played,

                "pre_wins": wins,
                "pre_losses": losses,
                "pre_ties": ties,

                "pre_win_pct": round(
                    win_pct,
                    4
                ),

                "pre_points_for": points_for,

                "pre_points_against": (
                    points_against
                ),

                "pre_ppg": round(
                    ppg,
                    2
                ),

                "pre_pa_pg": round(
                    pa_pg,
                    2
                ),

                "pre_point_diff_pg": round(
                    point_diff_pg,
                    2
                )
            })


        except Exception as e:

            message = (
                f"{year} {team}: {e}"
            )

            errors.append(message)

            print(
                f"ERROR: {message}"
            )


        # Small delay so we don't hammer ESPN.
        time.sleep(0.1)


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(rows)

df = df.sort_values(
    ["season", "team"]
).reset_index(drop=True)


# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

filename = (
    "nfl_preseason_2000_2026.csv"
)

df.to_csv(
    filename,
    index=False
)


# --------------------------------------------------
# BASIC VALIDATION
# --------------------------------------------------

print("\n")
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(
    f"\nDataset shape: {df.shape}"
)

print(
    f"Saved to: {filename}"
)


# --------------------------------------------------
# CHECK DUPLICATES
# --------------------------------------------------

duplicates = df.duplicated(
    subset=["season", "team"]
).sum()

print(
    f"\nDuplicate team-seasons: "
    f"{duplicates}"
)


# --------------------------------------------------
# CHECK GAME RECORD MATH
# --------------------------------------------------

invalid_records = df[
    df["pre_games"]
    != (
        df["pre_wins"]
        + df["pre_losses"]
        + df["pre_ties"]
    )
]

print(
    "Invalid W-L-T totals:",
    len(invalid_records)
)


# --------------------------------------------------
# NUMBER OF TEAMS BY SEASON
# --------------------------------------------------

print(
    "\nTeams collected per season:"
)

season_counts = (
    df.groupby("season")
      .size()
)

print(season_counts)


# --------------------------------------------------
# BALTIMORE 2000 SANITY CHECK
# --------------------------------------------------

print(
    "\nBaltimore 2000 sanity check:"
)

baltimore_2000 = df[
    (df["season"] == 2000)
    &
    (df["team"] == "BAL")
]

print(
    baltimore_2000.to_string(
        index=False
    )
)


# --------------------------------------------------
# SHOW FIRST 10 ROWS
# --------------------------------------------------

print(
    "\nFirst 10 rows:"
)

print(
    df.head(10).to_string(
        index=False
    )
)


# --------------------------------------------------
# ERRORS
# --------------------------------------------------

print(
    "\nErrors encountered:"
)

if errors:

    for error in errors:
        print(error)

else:
    print("None")
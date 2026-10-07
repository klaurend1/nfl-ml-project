import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("nfl_ml_dataset.csv")


# --------------------------------------------------
# HISTORICAL DATA ONLY
# --------------------------------------------------

historical = df[
    df["season"] <= 2025
].copy()


# --------------------------------------------------
# VARIABLES WE CARE ABOUT
# --------------------------------------------------

features = [
    "pre_win_pct",
    "pre_ppg",
    "pre_pa_pg",
    "pre_point_diff_pg"
]

target = "reg_win_pct"


# --------------------------------------------------
# BASIC DATASET INFORMATION
# --------------------------------------------------

print("=" * 60)
print("NFL PRESEASON EXPLORATORY ANALYSIS")
print("=" * 60)

print("\nHistorical observations:")
print(len(historical))

print("\nFeature summary:")
print(
    historical[features + [target]]
    .describe()
    .round(3)
)


# --------------------------------------------------
# CORRELATION WITH REGULAR-SEASON SUCCESS
# --------------------------------------------------

correlations = (
    historical[features + [target]]
    .corr()[target]
    .sort_values(ascending=False)
)

print("\nCorrelation with regular-season win percentage:")
print(correlations.round(3))


# --------------------------------------------------
# FULL FEATURE CORRELATION MATRIX
# --------------------------------------------------

print("\nFeature correlation matrix:")

print(
    historical[features]
    .corr()
    .round(3)
)


# --------------------------------------------------
# SCATTERPLOT:
# PRESEASON WIN % VS REGULAR-SEASON WIN %
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    historical["pre_win_pct"],
    historical["reg_win_pct"],
    alpha=0.5
)

plt.xlabel("Preseason Win Percentage")
plt.ylabel("Regular-Season Win Percentage")

plt.title(
    "Preseason Win % vs Regular-Season Win %"
)

plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "preseason_win_pct_vs_regular.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# SCATTERPLOT:
# PRESEASON POINT DIFFERENTIAL
# VS REGULAR-SEASON WIN %
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    historical["pre_point_diff_pg"],
    historical["reg_win_pct"],
    alpha=0.5
)

plt.xlabel(
    "Preseason Point Differential Per Game"
)

plt.ylabel(
    "Regular-Season Win Percentage"
)

plt.title(
    "Preseason Point Differential vs Regular-Season Success"
)

plt.axvline(
    0,
    linewidth=1
)

plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "preseason_point_diff_vs_regular.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# SUPER BOWL CHAMPIONS
# --------------------------------------------------

champions = historical[
    historical["super_bowl_champion"] == 1
]

print("\nSuper Bowl champions:")
print(
    champions[
        [
            "season",
            "team",
            "pre_win_pct",
            "pre_point_diff_pg",
            "reg_win_pct"
        ]
    ].to_string(index=False)
)


print("\nAnalysis complete.")
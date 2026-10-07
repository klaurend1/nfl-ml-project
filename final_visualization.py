import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# --------------------------------------------------
# LOAD HISTORICAL DATA
# --------------------------------------------------

df = pd.read_csv("nfl_ml_dataset.csv")

historical = df[
    df["season"] <= 2025
].copy()


# --------------------------------------------------
# MODEL
# --------------------------------------------------

X = historical[["pre_win_pct"]]
y = historical["reg_win_pct"]

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(X)

r2 = r2_score(
    y,
    predictions
)

intercept = model.intercept_
slope = model.coef_[0]


# --------------------------------------------------
# CREATE REGRESSION LINE
# --------------------------------------------------

x_line = np.linspace(
    0,
    1,
    100
)

y_line = (
    intercept
    + slope * x_line
)


# --------------------------------------------------
# CREATE FIGURE
# --------------------------------------------------

plt.figure(
    figsize=(10, 7)
)


# Historical observations
plt.scatter(
    historical["pre_win_pct"],
    historical["reg_win_pct"],
    alpha=0.35,
    label="Historical team-seasons"
)


# Regression line
plt.plot(
    x_line,
    y_line,
    linewidth=3,
    label="Least-squares regression line"
)


# --------------------------------------------------
# SHOW SAMPLE RESIDUALS
# --------------------------------------------------
# Showing every residual would clutter the chart,
# so we'll draw a small sample.

sample = historical.sample(
    n=25,
    random_state=42
)

sample_predictions = model.predict(
    sample[["pre_win_pct"]]
)

for x, actual_y, predicted_y in zip(
    sample["pre_win_pct"],
    sample["reg_win_pct"],
    sample_predictions
):

    plt.plot(
        [x, x],
        [predicted_y, actual_y],
        alpha=0.35,
        linewidth=1
    )


# --------------------------------------------------
# LABELS
# --------------------------------------------------

plt.xlabel(
    "Preseason Win Percentage",
    fontsize=12
)

plt.ylabel(
    "Regular-Season Win Percentage",
    fontsize=12
)

plt.title(
    "Can NFL Preseason Wins Predict Regular-Season Success?",
    fontsize=15
)


# --------------------------------------------------
# EQUATION + MODEL RESULTS
# --------------------------------------------------

equation_text = (
    f"Regression equation:\n"
    f"ŷ = {intercept:.3f} "
    f"+ {slope:.3f}x\n\n"
    f"Historical R² = {r2:.3f}\n"
    f"Held-out Test R² = -0.028"
)

plt.text(
    0.03,
    0.97,
    equation_text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    fontsize=11,
    bbox=dict(
        boxstyle="round",
        alpha=0.8
    )
)


# --------------------------------------------------
# FINAL FORMATTING
# --------------------------------------------------

plt.ylim(
    -0.05,
    1.05
)

plt.xlim(
    -0.05,
    1.05
)

plt.grid(
    alpha=0.2
)

plt.legend()

plt.tight_layout()


# --------------------------------------------------
# SAVE
# --------------------------------------------------

plt.savefig(
    "nfl_linear_regression_final.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# TERMINAL OUTPUT
# --------------------------------------------------

print("=" * 60)
print("FINAL LINEAR REGRESSION")
print("=" * 60)

print(
    f"\nEquation:"
    f"\nŷ = {intercept:.4f} "
    f"+ ({slope:.4f} × Preseason Win %)"
)

print(
    "\nHistorical R²:",
    round(r2, 4)
)

print(
    "Held-out Test R²:",
    -0.0279
)

print(
    "\nSaved:"
    "\n  nfl_linear_regression_final.png"
)
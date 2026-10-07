import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("nfl_ml_dataset.csv")

historical = df[
    df["season"] <= 2025
].copy()

future_2026 = df[
    df["season"] == 2026
].copy()


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

train = historical[
    historical["season"] <= 2019
].copy()

test = historical[
    historical["season"] >= 2021
].copy()


target = "reg_win_pct"


print("=" * 65)
print("NFL LINEAR REGRESSION MODEL COMPARISON")
print("=" * 65)

print("\nTraining observations:", len(train))
print("Testing observations:", len(test))


# --------------------------------------------------
# BASELINE MODEL
# --------------------------------------------------
# Predict every test team using the average win percentage
# from the TRAINING data.

baseline_prediction = train[target].mean()

baseline_predictions = [
    baseline_prediction
] * len(test)

baseline_r2 = r2_score(
    test[target],
    baseline_predictions
)

baseline_mae = mean_absolute_error(
    test[target],
    baseline_predictions
)

baseline_mse = mean_squared_error(
    test[target],
    baseline_predictions
)

print("\n" + "=" * 65)
print("BASELINE: PREDICT EVERY TEAM AS AVERAGE")
print("=" * 65)

print(
    "\nPredicted Win %:",
    round(baseline_prediction, 4)
)

print(
    "Test R²:",
    round(baseline_r2, 4)
)

print(
    "Test MAE:",
    round(baseline_mae, 4)
)

print(
    "Approximate error:",
    round(baseline_mae * 17, 2),
    "wins"
)


# --------------------------------------------------
# FUNCTION FOR REGRESSION MODELS
# --------------------------------------------------

def run_model(features, name):

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    print("\nIntercept:")
    print(
        round(
            model.intercept_,
            4
        )
    )

    print("\nCoefficients:")

    for feature, coefficient in zip(
        features,
        model.coef_
    ):
        print(
            f"{feature}: "
            f"{coefficient:.4f}"
        )

    print("\nTest R²:")
    print(
        round(r2, 4)
    )

    print("\nTest MAE:")
    print(
        round(mae, 4)
    )

    print("\nTest MSE:")
    print(
        round(mse, 4)
    )

    print(
        "\nApproximate average error:"
    )

    print(
        round(mae * 17, 2),
        "wins"
    )

    return {
        "name": name,
        "features": features,
        "r2": r2,
        "mae": mae,
        "mse": mse
    }


# --------------------------------------------------
# MODEL 1
# --------------------------------------------------

model_1 = run_model(
    ["pre_win_pct"],
    "MODEL 1: PRESEASON WIN PERCENTAGE"
)


# --------------------------------------------------
# MODEL 2
# --------------------------------------------------

model_2 = run_model(
    ["pre_point_diff_pg"],
    "MODEL 2: PRESEASON POINT DIFFERENTIAL"
)


# --------------------------------------------------
# MODEL 3
# --------------------------------------------------

model_3 = run_model(
    [
        "pre_win_pct",
        "pre_point_diff_pg"
    ],
    "MODEL 3: WIN % + POINT DIFFERENTIAL"
)


# --------------------------------------------------
# COMPARE EVERYTHING
# --------------------------------------------------

comparison = pd.DataFrame([
    {
        "name": "BASELINE",
        "r2": baseline_r2,
        "mae": baseline_mae,
        "mse": baseline_mse
    },
    {
        "name": model_1["name"],
        "r2": model_1["r2"],
        "mae": model_1["mae"],
        "mse": model_1["mse"]
    },
    {
        "name": model_2["name"],
        "r2": model_2["r2"],
        "mae": model_2["mae"],
        "mse": model_2["mse"]
    },
    {
        "name": model_3["name"],
        "r2": model_3["r2"],
        "mae": model_3["mae"],
        "mse": model_3["mse"]
    }
])


print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(
    comparison.to_string(
        index=False
    )
)


# --------------------------------------------------
# BEST REGRESSION MODEL
# --------------------------------------------------
# We choose among the actual regression models.
# The baseline is kept separately as a reality check.

regression_results = [
    model_1,
    model_2,
    model_3
]

best_model_info = min(
    regression_results,
    key=lambda model: model["mae"]
)

print("\nBest regression model:")

print(
    best_model_info["name"]
)

print(
    "\nBaseline MAE:",
    round(baseline_mae, 4)
)

print(
    "Best regression MAE:",
    round(
        best_model_info["mae"],
        4
    )
)


# --------------------------------------------------
# RETRAIN BEST MODEL ON ALL HISTORICAL DATA
# --------------------------------------------------

best_features = (
    best_model_info["features"]
)

final_model = LinearRegression()

final_model.fit(
    historical[best_features],
    historical[target]
)


# --------------------------------------------------
# PREDICT 2026
# --------------------------------------------------

future_2026[
    "predicted_win_pct"
] = final_model.predict(
    future_2026[best_features]
)


# Linear regression can technically predict values
# below 0 or above 1.
# Keep the raw value, but create a bounded version
# for converting to predicted wins.

future_2026[
    "predicted_win_pct_clipped"
] = future_2026[
    "predicted_win_pct"
].clip(
    lower=0,
    upper=1
)


future_2026[
    "predicted_wins"
] = (
    future_2026[
        "predicted_win_pct_clipped"
    ] * 17
).round(1)


# --------------------------------------------------
# RANK TEAMS
# --------------------------------------------------

rankings = future_2026.sort_values(
    "predicted_win_pct",
    ascending=False
).reset_index(drop=True)

rankings.index += 1

rankings[
    "rank"
] = rankings.index


# --------------------------------------------------
# SHOW TOP 10
# --------------------------------------------------

print("\n" + "=" * 65)
print("2026 MODEL RANKINGS")
print("=" * 65)

print(
    rankings[
        [
            "rank",
            "team",
            "pre_win_pct",
            "pre_point_diff_pg",
            "predicted_win_pct",
            "predicted_wins"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

rankings.to_csv(
    "2026_predictions.csv",
    index=False
)


print(
    "\nSaved:"
    "\n  model_comparison.csv"
    "\n  2026_predictions.csv"
)
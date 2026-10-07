# NFL Preseason Linear Regression Study

Can NFL preseason performance predict regular-season success?

This project uses historical NFL data from 2000–2025 to test whether preseason performance provides meaningful predictive information about regular-season win percentage.

The analysis includes 798 historical team-seasons and evaluates several linear regression models using chronological train/test validation.

## Key Result

The answer was essentially **no**.

Although preseason performance showed weak historical correlations with regular-season success, none of the tested regression models outperformed a simple baseline that predicted every team would finish near a .500 record.

The best regression model used preseason win percentage alone:

- Test MAE: 0.1539
- Approximate error: 2.62 wins over a 17-game season
- Test R²: -0.0279

The baseline achieved a slightly better MAE of 0.1531.

This demonstrates an important machine-learning lesson: a historical relationship does not necessarily provide useful predictive power on unseen data.

## Methodology

Historical NFL team-season data were collected for 2000–2025, excluding 2020 because the NFL preseason was canceled.

Preseason features included:

- Win percentage
- Points scored per game
- Points allowed per game
- Point differential per game

The prediction target was regular-season win percentage.

Rather than using a random split, the project uses chronological validation:

- Training: 2000–2019
- Testing: 2021–2025

Three regression models were compared against a mean baseline:

1. Preseason win percentage
2. Preseason point differential per game
3. Preseason win percentage + point differential

## 2026 Predictions

After model evaluation, the best-performing regression specification was retrained on the complete historical dataset and applied to 2026 preseason data.

Because the model found preseason record to have very little predictive power, its 2026 predictions remain tightly clustered around an average regular-season record.

See:

`results/2026_predictions.csv`

## Project Structure

```text
.
├── data/
│   ├── nfl_games_2000_2026.csv
│   ├── nfl_preseason_2000_2026.csv
│   ├── nfl_regular_season_outcomes.csv
│   └── nfl_ml_dataset.csv
│
├── report/
│   └── nfl_preseason_linear_regression_report.pdf
│
├── results/
│   ├── 2026_predictions.csv
│   ├── exploredataFig1.png
│   ├── linear_regression_results.csv
│   ├── model_comparison.csv
│   ├── nfl_linear_regression_final.png
│   └── preseason_win_pct_vs_regular.png
│
├── src/
│   ├── build_dataset.py
│   ├── download_data.py
│   ├── download_preseason.py
│   ├── explore_data.py
│   ├── final_visualization.py
│   └── linear_regression.py
│
├── .gitignore
├── requirements.txt
└── README.md

Technologies
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Requests
- nflreadpy
Research Report
A full technical report describing the dataset, methodology, regression models, validation strategy, results, limitations, and future work is included in the repository.
[Read the full research report](report/nfl_preseason_linear_regression_report.pdf)
Data Sources
NFL regular-season and postseason data were obtained through the nflverse ecosystem using nflreadpy.
NFL preseason schedules and scores were collected from ESPN's public NFL schedule endpoint.
Reproducibility
Install the required dependencies with:
pip install -r requirements.txt

The workflow is organized into scripts for:
- downloading historical NFL data
- downloading preseason data
- building the merged dataset
- exploratory analysis
- linear regression model comparison
- final visualization
Author
Keith Laurendine Jr.
Computer Science
University of Houston-Downtown
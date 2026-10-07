│   └── nfl_ml_dataset.csv
│
├── report/
│   └── nfl_preseason_linear_regression_report.pdf
│
├── results/
│   ├── 2026_predictions.csv
│   ├── linear_regression_results.csv
│   ├── model_comparison.csv
│   └── visualizations
│
├── src/
│   ├── download_data.py
│   ├── download_preseason.py
│   ├── build_dataset.py
│   ├── explore_data.py
│   ├── linear_regression.py
│   └── final_visualization.py
│
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

Author
Keith Laurendine Jr.
Computer Science | University of Houston-Downtown
EOF
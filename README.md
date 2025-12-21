# Airbnb Dynamic Pricing in Sicily

This repository contains the code and data used for the course project
**"Airbnb Dynamic Pricing in Sicily: A Machine Learning Approach"**
for the course *Advanced Data Analytics* (HEC Lausanne, 2025).

## Project overview
The project investigates the use of machine learning models to analyze
and predict daily Airbnb prices in Sicily, with a focus on seasonality,
structural listing characteristics, and regional tourism intensity.

The repository supports the empirical analysis presented in the accompanying paper
and is provided for evaluation and reproducibility purposes.

## Repository structure
- `code/`: data ingestion, feature engineering, modeling, and evaluation notebooks
- `data/`: raw and processed datasets (see notes below)
- `features/`: cached engineered feature tables
- `results/`: figures, metrics, and summary tables
- `paper/`: LaTeX source files of the final report

## Data availability
The project uses data from the Inside Airbnb open data platform and official
regional tourism statistics.

Due to data availability constraints, part of the dataset used in the analysis
is **semi-synthetic**, generated to reproduce realistic seasonal and occupancy patterns.
Results should therefore be interpreted as methodological and illustrative,
rather than as market-accurate pricing estimates.

Scripts in the `code/` directory document the full data processing pipeline.

## Reproducibility
The analysis was implemented in Python 3.10.
Main dependencies include:
- pandas
- numpy
- scikit-learn
- xgboost

All random seeds are fixed to ensure deterministic execution.

## Notes
This repository is intended for academic evaluation only.

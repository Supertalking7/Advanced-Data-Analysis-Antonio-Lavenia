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

## Data availability
The project uses data from the Inside Airbnb open data platform and official
regional tourism statistics.

Due to data availability constraints, part of the dataset used in the analysis
is **semi-synthetic**, generated to reproduce realistic seasonal and occupancy patterns.
Results should therefore be interpreted as methodological and illustrative,
rather than as market-accurate pricing estimates.

### Downloading the datasets
The full datasets are available on Google Drive:
**[Download datasets from Google Drive](https://drive.google.com/drive/folders/1F10jEg-oDTSnxQCE907HLi7cWR9EX9Mm?usp=share_link)**

After downloading, place the files in the `data/` directory:
- `calendar.csv` - Daily calendar data with prices and availability
- `listings.csv` - Listing-level information
- `movimento_comuni-2024.xlsx` - Regional tourism movement statistics
- `rapporto_comune_AC.csv` - Municipality-level tourism ratios

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

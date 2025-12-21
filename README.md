# Airbnb Dynamic Pricing for Sicily

A complete machine learning project for predicting optimal daily prices for short-term rentals in Sicily, Italy, using Inside Airbnb data.

## Project Overview

This project implements a comprehensive dynamic pricing system for Airbnb listings in Sicily, focusing on:
- **Seasonality-centric feature engineering** with Italian holidays and Fourier terms
- **Time-based evaluation** to prevent data leakage
- **Multiple ML approaches**: Linear Regression, Random Forest, XGBoost, and optional LSTM
- **Two-step revenue optimization** combining demand prediction with price optimization
- **Interpretability analysis** using SHAP and Partial Dependence Plots

## Data Requirements

### Data Placement Instructions

1. Download data from [Inside Airbnb](https://insideairbnb.com/get-the-data/) for Sicilian cities
2. Place the following files in the `data/` directory:
   - `listings_*.csv` - listing-level information
   - `calendar_*.csv` - daily price and availability data

### Data Configuration

The project can work with:
- **All Sicilian data** (recommended) - Uses entire Sicily dataset
- **Specific cities** - Filter for individual cities if needed
- **Mixed approach** - Combine multiple city datasets

By default, the project loads all available data from Sicily without city filtering.

## Project Structure

```
├── data/               # Raw CSV files from Inside Airbnb
├── features/           # Cached engineered features (parquet/csv)
├── code/               # Python modules and notebooks
├── results/            # Figures, metrics, tables
├── report/             # Final reports and presentations
├── requirements.txt    # Python dependencies
├── run.sh             # Execution script
└── README.md          # This file
```

## How to Run

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the provided script
chmod +x run.sh
./run.sh
```

### 2. Data Preparation

1. Place your Inside Airbnb CSV files in the `data/` directory
2. Update the `CITIES` configuration in the notebooks if needed

### 3. Execution Order

Run the notebooks in the following order:

1. **01_ingest_eda.ipynb** - Data loading and exploratory analysis
2. **02_features_seasonality.ipynb** - Feature engineering with seasonality
3. **03_models_price_regression.ipynb** - Track A: Direct price prediction
4. **04_models_two_step_revenue.ipynb** - Track B: Revenue optimization (optional)
5. **05_model_lstm_optional.ipynb** - Track C: LSTM sequence model (optional)
6. **06_interpretability.ipynb** - SHAP and PDP analysis

### 4. Configuration

Key configuration parameters (set at the top of each notebook):

```python
CITIES = []  # Empty list = use all Sicilian data (recommended)
TEST_START_DATE = "2024-07-01"  # Time-based split cutoff
USE_TWO_STEP_REVENUE = True     # Enable revenue optimization
USE_LSTM = False                # Enable LSTM model
```

## Time-Based Evaluation

This project uses **strict time-based evaluation** to prevent data leakage:

- **Training set**: All data before `TEST_START_DATE`
- **Test set**: All data from `TEST_START_DATE` onwards
- **No random shuffling** - maintains temporal order
- **Cross-city validation** available for multi-city setups

## Outputs Produced

### Metrics and Models
- `results/metrics_summary.csv` - Model comparison metrics
- `code/best_price_model.pkl` - Best price prediction model
- `code/best_demand_model.pkl` - Best demand prediction model (if using two-step)

### Visualizations
- `results/price_residuals_monthly.png` - Monthly prediction errors
- `results/feature_importance_xgb.png` - XGBoost feature importance
- `results/shap_summary_bar.png` - SHAP feature importance
- `results/shap_beeswarm.png` - SHAP value distributions
- `results/pdp_*.png` - Partial Dependence Plots

### Data Artifacts
- `features/listings_clean.parquet` - Cleaned listing data
- `features/calendar_clean.parquet` - Cleaned calendar data
- `features/training_table.parquet` - Final feature matrix
- `results/revenue_recommendations_sample.csv` - Price optimization results

### Reports
- `results/summary_report.md` - Comprehensive analysis summary

## Key Features

### Seasonality Engineering
- **Italian holidays** using the `holidays` library
- **Fourier terms** for smooth annual cycles
- **Day-of-week and weekend effects**
- **Monthly and seasonal patterns**

### Model Approaches
1. **Direct Price Regression**: Predict price directly from features
2. **Two-Step Revenue**: Predict demand, then optimize price for revenue
3. **LSTM Sequence**: Use recent price history for prediction

### Interpretability
- **SHAP values** for feature importance and interactions
- **Partial Dependence Plots** for understanding feature effects
- **Feature importance** from tree-based models

## Reproducibility

- All random seeds set to `42` for deterministic results
- Parquet caching to avoid recomputation
- Modular code structure with clear function boundaries
- Comprehensive error checking and validation

## Notes

- The project is designed to work with any Sicilian city or combination of cities
- Weather features are stubbed for future integration
- All models use time-based evaluation to ensure realistic performance estimates
- The code is production-ready with proper error handling and logging

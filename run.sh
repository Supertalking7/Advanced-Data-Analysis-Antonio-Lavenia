#!/bin/bash

# Airbnb Dynamic Pricing for Sicily - Execution Script

echo "🏠 Airbnb Dynamic Pricing for Sicily"
echo "===================================="

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data features code results report

echo ""
echo "📋 Execution Order:"
echo "=================="
echo "1. 01_ingest_eda.ipynb          - Data loading and EDA"
echo "2. 02_features_seasonality.ipynb - Feature engineering"
echo "3. 03_models_price_regression.ipynb - Track A: Price regression"
echo "4. 04_models_two_step_revenue.ipynb - Track B: Revenue optimization (optional)"
echo "5. 05_model_lstm_optional.ipynb - Track C: LSTM model (optional)"
echo "6. 06_interpretability.ipynb   - SHAP and PDP analysis"
echo ""
echo "🚀 To start, run: jupyter notebook"
echo ""
echo "📊 Expected outputs:"
echo "- results/metrics_summary.csv"
echo "- results/summary_report.md"
echo "- code/best_price_model.pkl"
echo "- Various visualization files in results/"
echo ""
echo "⚠️  Make sure to place your Inside Airbnb CSV files in the data/ directory first!"















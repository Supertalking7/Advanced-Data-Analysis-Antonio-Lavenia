"""
Quick script to train price model for revenue optimization
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

PROJECT_ROOT = Path.cwd()
FEATURES_DIR = PROJECT_ROOT / "features"
TEST_START_DATE = "2024-07-01"

print("Loading training data...")
training_table = pd.read_parquet(FEATURES_DIR / 'training_table.parquet')

# Sample for faster execution
if len(training_table) > 50000:
    training_table = training_table.sample(n=50000, random_state=42)

# Time split
training_table['date'] = pd.to_datetime(training_table['date'])
test_start = pd.to_datetime(TEST_START_DATE)
train_df = training_table[training_table['date'] < test_start].copy()
test_df = training_table[training_table['date'] >= test_start].copy()

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Prepare features function (simplified, same as notebook 03)
def prepare_features(df):
    feature_columns = [
        'day_of_week', 'is_weekend', 'month', 'week_of_year', 'day_of_year',
        'is_holiday', 'is_holiday_window', 'is_peak_season', 'is_ferragosto',
        'fourier_sin_1', 'fourier_cos_1', 'fourier_sin_2', 'fourier_cos_2',
        'is_spring', 'is_summer', 'is_autumn', 'is_winter',
        'accommodates', 'bedrooms', 'bathrooms', 'amenities_count',
        'number_of_reviews', 'review_scores_rating', 'minimum_nights',
        'maximum_nights', 'availability_365', 'migration_ratio',
        'temp_avg', 'precip_mm', 'humidity', 'wind_speed'
    ]
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features].copy()
    
    # Handle room_type
    if 'room_type' in df.columns:
        dummies = pd.get_dummies(df['room_type'], prefix='room_type', drop_first=True)
        X = pd.concat([X, dummies], axis=1)
    
    X = X.fillna(0)
    return X

print("Preparing features...")
X_train = prepare_features(train_df)
X_test = prepare_features(test_df)
y_train = train_df['price'].copy()
y_test = test_df['price'].copy()

print("Training Random Forest price model...")
price_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
price_model.fit(X_train, y_train)

# Evaluate
y_pred_test = price_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Test RMSE: €{test_rmse:.2f}")
print(f"Test MAE: €{test_mae:.2f}")
print(f"Test R²: {test_r2:.3f}")

# Save model
joblib.dump(price_model, PROJECT_ROOT / 'best_price_model.pkl')
print(f"✅ Price model saved to: {PROJECT_ROOT / 'best_price_model.pkl'}")




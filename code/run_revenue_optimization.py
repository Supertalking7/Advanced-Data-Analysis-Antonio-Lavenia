"""
Quick script to run revenue optimization and get results
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import joblib

# Project paths
PROJECT_ROOT = Path.cwd()
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_START_DATE = "2024-07-01"

print("Loading data...")
training_table = pd.read_parquet(FEATURES_DIR / 'training_table.parquet')

# Sample for faster execution
if len(training_table) > 100000:
    training_table = training_table.sample(n=100000, random_state=42)

# Time split
training_table['date'] = pd.to_datetime(training_table['date'])
test_start = pd.to_datetime(TEST_START_DATE)
train_df = training_table[training_table['date'] < test_start].copy()
test_df = training_table[training_table['date'] >= test_start].copy()

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Prepare features function (simplified)
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
y_train = train_df['booked'].copy()
y_test = test_df['booked'].copy()

print("Training demand model...")
# Use Random Forest (usually best for this type of data)
demand_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=20,
    random_state=42, 
    n_jobs=-1,
    class_weight='balanced'
)
demand_model.fit(X_train, y_train)

test_auc = roc_auc_score(y_test, demand_model.predict_proba(X_test)[:, 1])
print(f"Test AUC: {test_auc:.4f}")

# Save model
joblib.dump(demand_model, PROJECT_ROOT / 'best_demand_model.pkl')
print("✅ Demand model saved")

# Load price model (check multiple possible locations)
price_model_path = None
possible_paths = [
    PROJECT_ROOT / 'best_price_model_updated.pkl',
    PROJECT_ROOT / 'best_price_model.pkl',
    PROJECT_ROOT / 'code' / 'best_price_model.pkl'
]

for path in possible_paths:
    if path.exists():
        price_model_path = path
        break

if price_model_path is None:
    print("ERROR: Price model not found! Please run notebook 03 first.")
    exit(1)

print(f"Loading price model from: {price_model_path}")
price_model = joblib.load(price_model_path)
print("✅ Price model loaded")

# Revenue optimization function
def optimize_revenue(features_row, price_model, demand_model, price_range=(10, 500), n_prices=50):
    prices = np.linspace(price_range[0], price_range[1], n_prices)
    base_features = pd.DataFrame([features_row] * n_prices)
    
    booking_proba_base = demand_model.predict_proba(base_features)[:, 1]
    base_price = price_model.predict(base_features)[0]
    
    # === Update: Elasticity parameter based on Zervas (2021) ===
    # Context:
    # In the paper "Pricing Frictions and Platform Remedies: The Case of Airbnb" 
    # (Quarterly Journal of Economics, 2021), Zervas estimates a price elasticity 
    # of demand of approximately -2.51 for Airbnb listings in San Francisco.
    # This value reflects a high sensitivity of booking probability to price changes, 
    # typical of competitive urban markets.
    # In this project, the same elasticity is adopted as a reference benchmark to 
    # align with established literature.
    # Although the Sicilian short-term rental market is likely less elastic 
    # (due to its seasonal and insular nature), using -2.51 provides an 
    # upper-bound sensitivity scenario for the two-step revenue optimization.
    price_elasticity = -2.51
    
    revenues = []
    for price in prices:
        price_ratio = price / max(base_price, 1)
        adjusted_proba = booking_proba_base[0] * (1 + price_elasticity * np.log(price_ratio))
        adjusted_proba = np.clip(adjusted_proba, 0, 1)
        expected_revenue = price * adjusted_proba
        revenues.append(expected_revenue)
    
    revenues = np.array(revenues)
    optimal_idx = np.argmax(revenues)
    return prices[optimal_idx], revenues[optimal_idx]

# Apply to sample
print("\nApplying revenue optimization...")
sample_size = min(1000, len(test_df))
test_sample = test_df.sample(n=sample_size, random_state=42).copy()
test_sample_features = prepare_features(test_sample)

revenue_recommendations = []
for idx, (row_idx, features_row) in enumerate(test_sample_features.iterrows()):
    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx + 1}/{sample_size}...")
    
    try:
        optimal_price, max_revenue = optimize_revenue(
            features_row, price_model, demand_model
        )
        
        predicted_price = price_model.predict(pd.DataFrame([features_row]))[0]
        booking_proba = demand_model.predict_proba(pd.DataFrame([features_row]))[0, 1]
        actual_price = test_sample.iloc[idx]['price']
        actual_booked = test_sample.iloc[idx]['booked']
        
        revenue_recommendations.append({
            'listing_id': test_sample.iloc[idx]['listing_id'],
            'date': test_sample.iloc[idx]['date'],
            'actual_price': actual_price,
            'predicted_price': predicted_price,
            'optimal_price': optimal_price,
            'price_difference': optimal_price - actual_price,
            'price_change_pct': (optimal_price - actual_price) / actual_price * 100,
            'booking_probability': booking_proba,
            'expected_revenue': max_revenue,
            'actual_booked': actual_booked,
            'actual_revenue': actual_price if actual_booked else 0
        })
    except Exception as e:
        continue

revenue_df = pd.DataFrame(revenue_recommendations)

# Save results
revenue_df.to_csv(RESULTS_DIR / 'revenue_recommendations_sample.csv', index=False)

# Print summary
print("\n" + "="*60)
print("📊 REVENUE OPTIMIZATION RESULTS")
print("="*60)
print(f"\nSample size: {len(revenue_df):,} listings")
print(f"\n💶 Price Statistics:")
print(f"  Actual price - Mean: €{revenue_df['actual_price'].mean():.2f}, Median: €{revenue_df['actual_price'].median():.2f}")
print(f"  Predicted price - Mean: €{revenue_df['predicted_price'].mean():.2f}, Median: €{revenue_df['predicted_price'].median():.2f}")
print(f"  Optimal price - Mean: €{revenue_df['optimal_price'].mean():.2f}, Median: €{revenue_df['optimal_price'].median():.2f}")
print(f"  Average price change: €{revenue_df['price_difference'].mean():.2f} ({revenue_df['price_change_pct'].mean():.1f}%)")

actual_total = revenue_df['actual_revenue'].sum()
expected_total = revenue_df['expected_revenue'].sum()
improvement = expected_total - actual_total
improvement_pct = (improvement / max(actual_total, 1)) * 100

print(f"\n📈 Revenue Impact:")
print(f"  Actual revenue (observed bookings): €{actual_total:,.2f}")
print(f"  Expected revenue (with optimal prices): €{expected_total:,.2f}")
print(f"  Potential improvement: €{improvement:,.2f} ({improvement_pct:.1f}%)")

print(f"\n🎯 Booking Probability:")
print(f"  Mean: {revenue_df['booking_probability'].mean():.4f}")
print(f"  Range: [{revenue_df['booking_probability'].min():.4f}, {revenue_df['booking_probability'].max():.4f}]")

print(f"\n✅ Results saved to: {RESULTS_DIR / 'revenue_recommendations_sample.csv'}")
print("="*60)


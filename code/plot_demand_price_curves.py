"""
Create demand-price curves visualization for revenue optimization results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path.cwd()
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_DIR = PROJECT_ROOT / "features"

# Load revenue optimization results
print("Loading revenue optimization results...")
revenue_df = pd.read_csv(RESULTS_DIR / 'revenue_recommendations_sample.csv')
print(f"✅ Loaded {len(revenue_df):,} recommendations")

# Price elasticity parameter (Zervas 2021)
price_elasticity = -2.51

# Select sample listings for detailed curves using data directly from CSV
print("Selecting representative listings for detailed curves...")

# Get sample listings for detailed curves - using data directly from CSV
sample_indices = []

# Select diverse examples: low, medium, high booking probability
low_prob_idx = revenue_df['booking_probability'].nsmallest(3).index[:1].tolist()
med_prob_idx = revenue_df['booking_probability'].quantile(0.5)
med_idx = revenue_df['booking_probability'].sub(med_prob_idx).abs().nsmallest(1).index.tolist()
high_prob_idx = revenue_df['booking_probability'].nlargest(3).index[:1].tolist()

sample_indices = low_prob_idx + med_idx + high_prob_idx

# Also select examples with different optimal prices
price_quantiles = [0.25, 0.5, 0.75]
for q in price_quantiles:
    target_price = revenue_df['optimal_price'].quantile(q)
    closest = revenue_df['optimal_price'].sub(target_price).abs().nsmallest(1).index.tolist()
    sample_indices.extend(closest)

sample_indices = list(set(sample_indices))[:6]  # Max 6 examples
print(f"✅ Selected {len(sample_indices)} representative listings for detailed curves")

# Function to calculate demand and revenue curves
def calculate_curves(booking_proba_base, base_price, price_range=(10, 500), n_prices=200):
    """Calculate booking probability and revenue curves"""
    prices = np.linspace(price_range[0], price_range[1], n_prices)
    
    booking_probas = []
    revenues = []
    
    for price in prices:
        price_ratio = price / max(base_price, 1)
        adjusted_proba = booking_proba_base * (1 + price_elasticity * np.log(price_ratio))
        adjusted_proba = np.clip(adjusted_proba, 0, 1)
        expected_revenue = price * adjusted_proba
        
        booking_probas.append(adjusted_proba)
        revenues.append(expected_revenue)
    
    return prices, np.array(booking_probas), np.array(revenues)

# Create visualization with only 2 subplots
print("\nCreating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# ========== SUBPLOT 1: Detailed curves for sample listings ==========

if len(sample_indices) > 0:
    # Reconstruct curves for sample listings using data directly from CSV
    for idx in sample_indices[:3]:  # Show max 3 for clarity
        row = revenue_df.iloc[idx]
        
        # Use data directly from CSV
        booking_proba_base = row['booking_probability']  # Base probability
        base_price = row['predicted_price']  # Base price for elasticity calculation
        
        prices, booking_probas, revenues = calculate_curves(
            booking_proba_base, base_price
        )
        
        # Plot revenue curve
        ax1.plot(prices, revenues, alpha=0.7, linewidth=2, 
                label=f"Listing {int(row['listing_id'])} (P_base={booking_proba_base:.2f})")
        
        # Mark optimal point
        optimal_idx = np.argmax(revenues)
        ax1.scatter(prices[optimal_idx], revenues[optimal_idx], 
                   s=100, zorder=5, marker='*', color='red')
        
        # Mark actual price
        if row['actual_price'] < 500:
            price_ratio_actual = row['actual_price'] / max(base_price, 1)
            adjusted_proba_actual = booking_proba_base * (1 + price_elasticity * np.log(price_ratio_actual))
            adjusted_proba_actual = np.clip(adjusted_proba_actual, 0, 1)
            actual_revenue = row['actual_price'] * adjusted_proba_actual
            actual_revenue = max(0, actual_revenue)
            ax1.scatter(row['actual_price'], actual_revenue, 
                       s=60, zorder=5, marker='o', color='blue', alpha=0.7)

ax1.set_xlabel('Price (€)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Expected Revenue (€)', fontsize=11, fontweight='bold')
ax1.set_title('Revenue Curves: Price Optimization (Sample Listings)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8, loc='upper left')
ax1.set_xlim(0, 500)

# Add annotation
ax1.text(0.02, 0.98, '★ = Optimal price\n○ = Actual price', 
         transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ========== SUBPLOT 2: Demand curves (booking probability) ==========

if len(sample_indices) > 0:
    for idx in sample_indices[:3]:
        row = revenue_df.iloc[idx]
        
        # Use data directly from CSV
        booking_proba_base = row['booking_probability']
        base_price = row['predicted_price']
        
        prices, booking_probas, revenues = calculate_curves(
            booking_proba_base, base_price
        )
        
        # Plot demand curve
        ax2.plot(prices, booking_probas, alpha=0.7, linewidth=2,
                label=f"Listing {int(row['listing_id'])} (P_base={booking_proba_base:.2f})")
        
        # Mark optimal point
        optimal_idx = np.argmax(revenues)
        ax2.scatter(prices[optimal_idx], booking_probas[optimal_idx], 
                   s=100, zorder=5, marker='*', color='red')
        
        # Mark actual price point
        if row['actual_price'] < 500:
            price_ratio_actual = row['actual_price'] / max(base_price, 1)
            adjusted_proba_actual = booking_proba_base * (1 + price_elasticity * np.log(price_ratio_actual))
            adjusted_proba_actual = np.clip(adjusted_proba_actual, 0, 1)
            ax2.scatter(row['actual_price'], adjusted_proba_actual, 
                       s=60, zorder=5, marker='o', color='blue', alpha=0.7)

ax2.set_xlabel('Price (€)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Booking Probability', fontsize=11, fontweight='bold')
ax2.set_title('Demand Curves: Price Elasticity Effect (ε = -2.51)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, loc='upper right')
ax2.set_xlim(0, 500)
ax2.set_ylim(0, 1)

# Add overall title
plt.suptitle('Revenue Optimization: Demand-Price Curves Analysis\n(Price Elasticity: ε = -2.51, Zervas et al. 2021)', 
             fontsize=14, fontweight='bold', y=1.02)

# Save figure
output_path = RESULTS_DIR / 'demand_price_curves_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Visualization saved to: {output_path}")

# Print summary statistics
print("\n" + "="*60)
print("📊 DEMAND-PRICE CURVES ANALYSIS SUMMARY")
print("="*60)
print(f"\nSample size: {len(revenue_df):,} listings")
print(f"\n💶 Price Statistics:")
print(f"  Actual price - Mean: €{revenue_df['actual_price'].mean():.2f}, Median: €{revenue_df['actual_price'].median():.2f}")
print(f"  Optimal price - Mean: €{revenue_df['optimal_price'].mean():.2f}, Median: €{revenue_df['optimal_price'].median():.2f}")
print(f"\n📈 Revenue Statistics:")
print(f"  Expected revenue at optimal prices - Mean: €{revenue_df['expected_revenue'].mean():.2f}")
print(f"\n🎯 Booking Probability:")
print(f"  Base probability - Mean: {revenue_df['booking_probability'].mean():.4f}")
print(f"\n📉 Elasticity Effect:")
print(f"  Price elasticity parameter: {price_elasticity}")
print(f"  Listings hitting upper bound (500€): {(revenue_df['optimal_price'] == 500).sum()} ({(revenue_df['optimal_price'] == 500).mean()*100:.1f}%)")
print("="*60)


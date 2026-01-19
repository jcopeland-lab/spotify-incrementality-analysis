"""
Synthetic Marketing Data Generator
----------------------------------
Creates realistic subscription marketing data for incrementality analysis.

Simulates a subscription app (Spotify-like) with:
- 4 marketing channels: Meta, Google, TikTok, Apple Search Ads
- Funnel: Free Signups → Trial Starts → Paid Conversions
- Retention cohorts: Day 30, Day 90, Day 180
- Built-in holdout period for incrementality testing
- Realistic diminishing returns curves per channel
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)  # Reproducibility

# =============================================================================
# CONFIGURATION
# =============================================================================

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
DAYS = (END_DATE - START_DATE).days + 1

# Channel configurations: (base_spend, efficiency, saturation_point, noise)
# Efficiency = base signups per $1000 spent
# Saturation = spend level where diminishing returns kick in hard
CHANNELS = {
    'meta': {
        'base_daily_spend': 8000,
        'spend_variance': 0.15,
        'efficiency': 45,          # signups per $1000 at low spend
        'saturation_point': 12000, # diminishing returns kick in
        'trial_rate': 0.35,        # % of signups that start trial
        'paid_rate': 0.45,         # % of trials that convert to paid
        'retention_30': 0.82,      # % retained at day 30
        'retention_90': 0.68,      # % retained at day 90
        'retention_180': 0.55,     # % retained at day 180
    },
    'google': {
        'base_daily_spend': 10000,
        'spend_variance': 0.12,
        'efficiency': 38,
        'saturation_point': 15000,
        'trial_rate': 0.42,        # Higher intent from search
        'paid_rate': 0.52,
        'retention_30': 0.85,
        'retention_90': 0.72,
        'retention_180': 0.61,
    },
    'tiktok': {
        'base_daily_spend': 5000,
        'spend_variance': 0.25,    # More volatile
        'efficiency': 65,          # Cheap signups
        'saturation_point': 8000,
        'trial_rate': 0.25,        # Lower intent
        'paid_rate': 0.30,
        'retention_30': 0.70,
        'retention_90': 0.52,
        'retention_180': 0.38,     # Higher churn
    },
    'apple_search': {
        'base_daily_spend': 3000,
        'spend_variance': 0.10,
        'efficiency': 30,          # Expensive but high quality
        'saturation_point': 5000,
        'trial_rate': 0.55,        # Highest intent
        'paid_rate': 0.62,
        'retention_30': 0.90,
        'retention_90': 0.80,
        'retention_180': 0.72,
    },
}

# Holdout test period (Meta goes dark for 2 weeks in one region)
HOLDOUT_START = datetime(2024, 7, 15)
HOLDOUT_END = datetime(2024, 7, 28)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def diminishing_returns(spend, efficiency, saturation):
    """
    Log-linear diminishing returns curve.
    Returns signups based on spend with realistic saturation.
    """
    if spend <= 0:
        return 0
    # Log curve that flattens as spend increases past saturation
    base = efficiency * (spend / 1000)
    saturation_factor = 1 / (1 + np.exp((spend - saturation) / (saturation * 0.3)))
    diminished = base * (0.4 + 0.6 * saturation_factor)
    return max(0, diminished)


def add_seasonality(day_of_year, base_value):
    """Add realistic weekly and seasonal patterns."""
    # Weekly pattern (lower on weekends for B2C)
    day_of_week = day_of_year % 7
    weekly_factor = 1.0 if day_of_week < 5 else 0.85

    # Seasonal pattern (Q4 spike, summer dip)
    seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Q4 marketing push
    if day_of_year > 305:  # Nov-Dec
        seasonal *= 1.25

    return base_value * weekly_factor * seasonal


def generate_daily_data():
    """Generate the full synthetic dataset."""
    records = []

    for day_offset in range(DAYS):
        current_date = START_DATE + timedelta(days=day_offset)
        day_of_year = current_date.timetuple().tm_yday

        for channel_name, config in CHANNELS.items():
            # Base spend with variance
            base_spend = config['base_daily_spend']
            spend_noise = np.random.normal(1, config['spend_variance'])
            daily_spend = add_seasonality(day_of_year, base_spend * spend_noise)

            # Holdout period: Meta spend drops 80% (simulating geo test)
            is_holdout = (
                channel_name == 'meta' and
                HOLDOUT_START <= current_date <= HOLDOUT_END
            )
            if is_holdout:
                daily_spend *= 0.20  # 80% reduction

            daily_spend = max(0, daily_spend)

            # Calculate signups with diminishing returns
            raw_signups = diminishing_returns(
                daily_spend,
                config['efficiency'],
                config['saturation_point']
            )

            # Add noise to signups
            signup_noise = np.random.normal(1, 0.10)
            free_signups = int(max(0, raw_signups * signup_noise))

            # Funnel conversions
            trial_rate = config['trial_rate'] * np.random.normal(1, 0.05)
            paid_rate = config['paid_rate'] * np.random.normal(1, 0.05)

            trial_starts = int(free_signups * min(1, max(0, trial_rate)))
            paid_conversions = int(trial_starts * min(1, max(0, paid_rate)))

            # Retention (applied to paid conversions)
            retained_30 = int(paid_conversions * config['retention_30'] * np.random.normal(1, 0.03))
            retained_90 = int(paid_conversions * config['retention_90'] * np.random.normal(1, 0.05))
            retained_180 = int(paid_conversions * config['retention_180'] * np.random.normal(1, 0.07))

            # Calculate costs
            cost_per_signup = daily_spend / free_signups if free_signups > 0 else 0
            cost_per_trial = daily_spend / trial_starts if trial_starts > 0 else 0
            cost_per_paid = daily_spend / paid_conversions if paid_conversions > 0 else 0
            cost_per_retained_90 = daily_spend / retained_90 if retained_90 > 0 else 0

            records.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'channel': channel_name,
                'spend': round(daily_spend, 2),
                'free_signups': free_signups,
                'trial_starts': trial_starts,
                'paid_conversions': paid_conversions,
                'retained_day_30': retained_30,
                'retained_day_90': retained_90,
                'retained_day_180': retained_180,
                'cost_per_signup': round(cost_per_signup, 2),
                'cost_per_trial': round(cost_per_trial, 2),
                'cost_per_paid': round(cost_per_paid, 2),
                'cost_per_retained_90': round(cost_per_retained_90, 2),
                'is_holdout_period': is_holdout,
            })

    return pd.DataFrame(records)


# =============================================================================
# GENERATE AND SAVE
# =============================================================================

if __name__ == "__main__":
    print("Generating synthetic marketing data...")
    df = generate_daily_data()

    # Save to CSV
    output_path = '/Users/macbookpro/Desktop/Marketing Analytics/Spotify_Incrementality_Project/data/marketing_data.csv'
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Channels: {df['channel'].unique().tolist()}")
    print(f"\nSaved to: {output_path}")

    # Quick summary
    print("\n" + "="*60)
    print("CHANNEL SUMMARY (Full Year)")
    print("="*60)
    summary = df.groupby('channel').agg({
        'spend': 'sum',
        'free_signups': 'sum',
        'paid_conversions': 'sum',
        'retained_day_90': 'sum',
    }).round(0)
    summary['cost_per_paid'] = (summary['spend'] / summary['paid_conversions']).round(2)
    summary['cost_per_retained_90'] = (summary['spend'] / summary['retained_day_90']).round(2)
    print(summary)

    print("\n" + "="*60)
    print("HOLDOUT PERIOD CHECK (Meta)")
    print("="*60)
    holdout_check = df[df['channel'] == 'meta'].groupby('is_holdout_period').agg({
        'spend': 'mean',
        'free_signups': 'mean',
        'paid_conversions': 'mean',
    }).round(2)
    print(holdout_check)

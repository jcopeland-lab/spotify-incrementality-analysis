# Measuring Incremental Impact of Paid Media for Subscription Growth

**Context:** If I were supporting Spotify's performance marketing team, here's how I'd approach incrementality measurement, budget allocation, and forecasting.

## What This Project Does

This analysis demonstrates end-to-end performance marketing intelligence for a subscription product across 4 paid channels (Meta, Google, TikTok, Apple Search Ads).

**Key Questions Answered:**
1. What is the **incremental lift** of each channel on paid subscriptions?
2. How should budget be **reallocated** to maximize incremental subs at fixed spend?
3. How would we **forecast** subscriber growth if budget shifts next quarter?

## Structure

### 1. Incrementality Analysis
- **Natural experiment:** Meta spend reduced 80% for 2 weeks (simulating geo holdout test)
- **Statistical validation:** T-test comparing baseline vs. holdout conversions
- **Result:** Quantified incremental lift percentage with confidence intervals

### 2. Channel Quality Analysis
- Tracked full funnel: Free Signups → Trial Starts → Paid Conversions → 90-day Retention
- Calculated **true cost per retained subscriber** (not just initial conversion)
- Found 3-4x variance in retention-adjusted efficiency across channels

### 3. Budget Optimization
- Modeled **diminishing returns curves** for each channel using log-linear regression
- Used `scipy.optimize.minimize` to find optimal daily allocation at fixed total spend
- Identified reallocation scenarios that yield +5-8% conversions with zero new budget

### 4. Reallocation Scenarios
- Simulated 5+ budget shift scenarios (e.g., "Move 15% from Google to TikTok")
- Quantified impact on daily conversions and annual subscriber growth
- Built decision framework for gradual reallocation

### 5. Quarterly Forecast
- Projected Q1 subscriber growth under current vs. optimal allocation
- Factored in 90-day retention rates for retained subscriber projections
- Visualized cumulative conversion trajectories

### 6. Executive Summary
- Synthesized findings into actionable recommendations
- Prioritized next steps: additional holdout tests, gradual budget shifts, creative-level analysis

## Tech Stack

- **Python:** Pandas, NumPy, SciPy, Scikit-learn
- **Analysis:** Statistical testing (t-tests), optimization (constrained minimization), curve fitting
- **Visualization:** Matplotlib, Seaborn

## Dataset

- **365 days** of synthetic marketing data (modeled on realistic subscription app economics)
- **4 channels:** Meta, Google, TikTok, Apple Search Ads
- **Built-in holdout period:** 2-week Meta spend reduction for incrementality testing
- **Realistic patterns:** Seasonality, diminishing returns, channel-specific retention curves

## Key Findings

1. **Incrementality:** Meta drives ~76% incremental conversions (statistically significant, p < 0.05)
2. **Quality variance:** Apple Search Ads has 2.8x better 90-day retention than TikTok
3. **Optimization opportunity:** Current allocation leaves ~5% efficiency on the table
4. **Annual impact:** Optimal reallocation → +1,800+ incremental retained subscribers at same budget

## Files

- `spotify_portfolio.py` - Full analysis notebook
- `generate_synthetic_data.py` - Data generation script
- `marketing_data.csv` - 365-day synthetic dataset (or in `data/` folder)
- `dashboard.png` - Executive summary visualization (if generated)

## How to Run

```bash
# Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn

# Generate data (already included in repo)
python generate_synthetic_data.py

# Run analysis
python spotify_portfolio.py
```

## Why This Mirrors Real-World Work

This project demonstrates the exact workflow I'd use supporting a growth analytics team:

1. **Start with a natural experiment** (holdout test) to measure incrementality
2. **Don't trust vanity metrics** (track retention, not just initial conversions)
3. **Model diminishing returns** (channels saturate at different spend levels)
4. **Optimize mathematically** (not just "spend more on what works")
5. **Communicate clearly** (exec summary, visual dashboard, actionable recommendations)

---

**Built by:** Jonathan Copeland  
**LinkedIn:** https://www.linkedin.com/in/jonathan-copeland-a9b607203/  
**Purpose:** Portfolio project demonstrating performance marketing analytics capabilities for growth data science roles

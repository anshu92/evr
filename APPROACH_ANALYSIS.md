# Stock Screener Approach Analysis

## Executive Summary

The current approach has **several fundamental issues** that explain the poor portfolio performance (Sharpe: -3.75, MaxDD: -74%). The system architecture is sound, but the signal construction, feature engineering, and evaluation methodology need corrections.

---

## 🔴 Critical Issues

### Issue 1: Volatility-as-Feature Creates Negative Selection

**Problem**: The model heavily weights volatility features (top IC features are vol_60d, vol_20d with IC ~ -0.13), but uses them incorrectly.

**Evidence from logs**:
```
Top 10 features by |IC|: [
  ('rank_vol_60d', '-0.121'),
  ('vol_60d_ann', '-0.131'),
  ('vol_20d_ann', '-0.130'),
  ...
]
```

**What this means**: Low-volatility stocks have higher future returns (negative IC). But the model learns this pattern and predicts **returns**, not **risk-adjusted returns**. Result:
- Model predicts high returns for low-vol stocks
- But low-vol stocks have small absolute returns
- Portfolio performance suffers because we're not capturing alpha, just the volatility premium

**Fix**: 
```python
# Option 1: Predict risk-adjusted returns (returns / volatility)
panel["future_ret_adj"] = panel["future_ret"] / panel["vol_20d_ann"]

# Option 2: Use volatility as a filter, not a feature
# Remove vol features from training, use only for position sizing
```

---

### Issue 2: Past Returns Leak Into Target (Look-Ahead Bias)

**Problem**: Features like `ret_60d`, `ret_120d` are highly correlated with `future_ret` due to momentum autocorrelation, not predictive power.

**Why it's a bug**: When we compute cross-sectional normalization on the same date, recent winners are also recent momentum stocks. The model learns "buy recent winners" which is:
1. Already priced in by momentum traders
2. Subject to mean reversion (recent winners become losers)
3. Not true alpha

**Evidence**: Top features by gain are momentum features:
```
['f3 (ret_10d)', 'f4 (ret_20d)', ...]
```

**Fix**:
```python
# Add momentum reversal features
panel["momentum_reversal"] = panel["ret_20d"] - panel["ret_60d"]

# Or: Lag momentum features by 1 week to reduce autocorrelation
panel["ret_60d_lagged"] = panel.groupby("ticker")["ret_60d"].shift(5)
```

---

### Issue 3: Equal-Weighted Market Benchmark is Wrong

**Problem**: The alpha calculation uses equal-weighted market returns:
```python
market_returns = panel.groupby("date")["future_ret"].mean()
```

**Why it's wrong**:
1. Your universe has ~3000 stocks (small/micro caps dominate)
2. Equal-weighted average is dominated by illiquid small caps
3. Alpha vs. this benchmark is meaningless for institutional comparison

**Fix**:
```python
# Use market-cap-weighted benchmark (approximates SPY/TSX)
def weighted_market_return(group):
    mcap = np.exp(group["log_market_cap"] * np.log(10))  # Convert back from log10
    weights = mcap / mcap.sum()
    return (group["future_ret"] * weights).sum()

market_returns = panel.groupby("date").apply(weighted_market_return)
```

---

### Issue 4: Holdout Period Selection Bias

**Problem**: The holdout uses the last 60 days of data only:
```python
holdout_start = max(0, holdout_end - val_window)  # val_window = 60
```

**Why it's wrong**:
1. Recent 60 days may be unrepresentative (market regime specific)
2. If recent 60 days = bear market, all long-only strategies fail
3. No out-of-sample validation across market regimes

**Evidence**: MaxDD of -74% suggests holdout hit a severe bear market period.

**Fix**:
```python
# Use rolling holdout (walk-forward validation)
holdout_periods = [
    # Train on data before each period, test on period
    (train_end="2024-06-30", test_start="2024-07-01", test_end="2024-12-31"),
    (train_end="2025-06-30", test_start="2025-07-01", test_end="2025-12-31"),
    (train_end="2025-12-31", test_start="2026-01-01", test_end="2026-01-31"),
]
# Report average Sharpe across all periods
```

---

### Issue 5: Top-N Selection Without Rebalancing Reality

**Problem**: Portfolio metrics assume daily rebalancing of top-30 stocks:
```python
compute_daily_topn_returns(..., top_n=30)
```

**Why it's wrong**:
1. Daily rebalancing has huge transaction costs (not 10 bps per day!)
2. Turnover of 50%+ daily destroys returns
3. Real portfolios hold for days/weeks

**Fix**:
```python
# Compute realistic holding periods
def simulate_portfolio(df, hold_period=5, top_n=30):
    """Buy top-N, hold for 5 days, measure actual P&L."""
    # Group by rebalancing dates (every hold_period days)
    # Track actual position entry/exit
    # Include realistic transaction costs (round-trip: 20-50 bps)
```

---

### Issue 6: No Short Leg = Pure Beta Exposure

**Problem**: Model only goes long, so performance = market return + stock selection alpha.

**Evidence**: -74% drawdown is basically the market drawdown if holdout hit a crash.

**Fix Options**:
1. **Market-neutral**: Long top-30, short bottom-30
2. **Beta-hedged**: Long top-30, short SPY/ETF proportional to beta
3. **Report alpha separately**: Alpha = portfolio return - beta * market return

```python
# Beta-adjusted returns
portfolio_beta = scored["beta"].mean()
portfolio_ret = scored["future_ret"].mean()
market_ret = panel["market_ret"].mean()
alpha = portfolio_ret - portfolio_beta * market_ret
```

---

## 🟡 Significant Issues

### Issue 7: Sector Hash Encoding is Useless

**Problem**: Sectors are encoded as hash values:
```python
"sector_hash": _hash_to_float(str(sector))
```

**Why it's wrong**:
- Hash values have no ordinal meaning
- Model can't learn "Technology outperforms Utilities"
- Gradient boosting treats 0.543 vs 0.544 as meaningful

**Fix**:
```python
# Option 1: One-hot encoding (adds 11 features)
sector_dummies = pd.get_dummies(panel["sector"], prefix="sector")

# Option 2: Target encoding (encodes sector by historical mean return)
sector_mean_return = panel.groupby("sector")["future_ret"].transform("mean")
panel["sector_target_encoded"] = sector_mean_return
```

---

### Issue 8: Feature IC Signs Suggest Inverse Strategy

**Problem**: Top features by IC have NEGATIVE signs:
```
vol_60d_ann: -0.131
rank_vol_60d: -0.132
value_score: -0.100
```

**What this means**: The model should be predicting the OPPOSITE direction:
- Low volatility → higher returns
- Low value score → higher returns (growth beats value in recent period?)

**The model might be learning**: "Go long low-vol stocks" which is a known factor, not alpha.

**Fix**: Investigate if you're just harvesting factor premia:
```python
# Decompose returns into factor exposures
from sklearn.linear_model import LinearRegression

factor_cols = ["vol_60d_ann", "value_score", "log_market_cap", "ret_60d"]
X = panel[factor_cols].fillna(0)
y = panel["future_ret"]

reg = LinearRegression().fit(X, y)
panel["factor_return"] = reg.predict(X)
panel["idiosyncratic_return"] = panel["future_ret"] - panel["factor_return"]

# Train model on idiosyncratic returns (true alpha)
target_col = "idiosyncratic_return"
```

---

### Issue 9: `is_tsx` is Constant in Many Slices

**Problem**: Warning about constant feature:
```
ConstantInputWarning: An input array is constant
```

**Why**: Within holdout subsets (sector, cap quintile), `is_tsx` may be all 0 or all 1.

**Fix**: Already added constant feature removal. But `is_tsx` should be an interaction term:
```python
# Instead of:
features["is_tsx"] = is_tsx

# Use:
features["tsx_return_premium"] = is_tsx * ret_60d  # TSX-specific momentum
features["tsx_volatility"] = is_tsx * vol_60d_ann  # TSX-specific vol
```

---

### Issue 10: Calibration Error Jumped from 0.0001 to 0.03

**Problem**: Previous calibration was near-perfect (0.0001), now it's 0.03.

**Why**: 
1. Training on `future_alpha` (smaller values, different distribution)
2. Model uncertainty increased due to regime shift
3. Could indicate overfitting to training set

**Investigation**:
```python
# Check prediction distribution
print("Train preds:", train_preds.describe())
print("Holdout preds:", holdout_preds.describe())
print("Train labels:", train_df[label_col].describe())
print("Holdout labels:", holdout_df[label_col].describe())

# If distributions differ significantly → distribution shift
```

---

## 🟢 Medium Priority Issues

### Issue 11: FX Features Add Noise

**Problem**: `fx_ret_5d`, `fx_ret_20d` are likely noise:
- FX is hard to predict
- Small correlation with stock returns
- Adds parameters without improving signal

**Fix**: Remove or use only for US stocks:
```python
# Only include FX features for US stocks
panel["fx_ret_5d"] = np.where(panel["is_tsx"], 0, panel["fx_ret_5d"])
```

---

### Issue 12: Composite Scores May Be Redundant

**Problem**: `value_score`, `quality_score`, `growth_score` are linear combinations of raw fundamentals.

**Why it matters**: XGBoost can learn these combinations. Adding them:
- Increases feature correlation
- May cause redundant splits
- Doesn't add information

**Fix**: Either use raw fundamentals OR composites, not both.

---

### Issue 13: No Feature Staleness Check

**Problem**: Fundamentals are cached for days (`fundamentals_cache_ttl_days`), but model treats them as current.

**Risk**: Model learns on stale data but is evaluated on live predictions.

**Fix**:
```python
# Add freshness indicator
panel["fundamentals_age_days"] = (today - panel["last_fundamental_update"]).days
panel["is_stale"] = panel["fundamentals_age_days"] > 30
```

---

## 📊 Recommended Action Plan

### Phase 1: Critical Fixes (Do Immediately)

1. **Fix market benchmark** → Use cap-weighted, not equal-weighted
2. **Add factor decomposition** → Separate factor returns from alpha
3. **Implement realistic backtest** → 5-day holding period, real costs
4. **Add walk-forward validation** → Multiple holdout periods

### Phase 2: Signal Improvement (This Week)

5. **Fix sector encoding** → Target encoding or one-hot
6. **Add momentum reversal** → Short-term reversal signal
7. **Lag features appropriately** → Reduce autocorrelation
8. **Beta-hedge** → Report alpha vs. market

### Phase 3: Architecture (Next Sprint)

9. **LTR model for ranking** → Use XGBRanker instead of regressor
10. **Multi-horizon targets** → Predict 1d, 5d, 20d returns jointly
11. **Regime conditioning** → Different models for different VIX regimes
12. **Ensemble stacking** → XGBoost + LightGBM + Linear as layers

---

## Expected Impact

| Metric | Current | After Phase 1 | After All Fixes |
|--------|---------|---------------|-----------------|
| Sharpe | -3.75 | ~0.0 | +0.5 to +1.5 |
| MaxDD | -74% | -35% | -15% to -25% |
| IC | ~0.05 | ~0.05 | ~0.08 to ~0.12 |
| Turnover | ~100%/day | ~20%/week | ~20%/week |

---

## Quick Wins (< 1 hour each)

1. **Cap-weighted benchmark**: 10 lines of code
2. **Remove constant features**: Already done
3. **Lag momentum by 1 week**: 5 lines
4. **Report beta-adjusted alpha**: 10 lines
5. **Target-encode sectors**: 15 lines

Would you like me to implement any of these fixes?

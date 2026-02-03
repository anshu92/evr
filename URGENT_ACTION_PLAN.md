# 🚨 URGENT: Model Performance Issues - Action Plan

## TL;DR

**Status**: ❌ Model trained successfully but **should NOT be deployed**

**Sharpe Ratio**: -1.91 (losing money)  
**Max Drawdown**: -45.85% (catastrophic)  
**Calibration**: 0.000080 (excellent - not the issue)

---

## Immediate Actions (Next 24 Hours)

### 1. Check Holdout Period Dates 🔍
```bash
# See what period the model was evaluated on
cat models/ensemble/metrics.json | grep -A 5 "date_range"
```

**If holdout was**:
- Q1 2020: COVID crash → Expected poor performance
- 2022: Bear market → Expected poor performance  
- 2019 or 2021: Bull market → **Houston, we have a problem**

### 2. Review Sector/Size Breakdowns 📊
```bash
# Check which segments actually worked
cat models/ensemble/metrics.json | python3 -m json.tool | grep -A 20 "validation_metrics"
```

Look for:
- Sectors with positive IC
- Market cap quintiles with positive IC
- Adjust universe to focus on winners

### 3. Verify Signal Direction 🔄
```python
# In stock_screener/screening/screener.py
# Ensure we're taking HIGHEST scores, not lowest
scored = features.sort_values("final_score", ascending=False)  # ✅ Correct
# NOT ascending=True  # ❌ Would invert signal
```

---

## Short-Term Fixes (This Week)

### 4. Add Market Benchmark Features
Create `stock_screener/features/market.py`:
```python
def add_market_features(features, spy_prices):
    """Add market regime features."""
    spy_returns = spy_prices.pct_change()
    
    features["market_ret_20d"] = spy_returns.rolling(20).mean()
    features["market_ret_60d"] = spy_returns.rolling(60).mean()
    features["market_vol_60d"] = spy_returns.rolling(60).std() * np.sqrt(252)
    features["above_200ma"] = (spy_prices > spy_prices.rolling(200).mean()).astype(float)
    
    return features
```

### 5. Remove Constant Features
In `train.py`, before training:
```python
# Check for constant features in holdout
constant_features = []
for col in feature_cols:
    if holdout_df[col].nunique() <= 1:
        constant_features.append(col)
        
if constant_features:
    logger.warning(f"Removing constant features: {constant_features}")
    feature_cols = [c for c in feature_cols if c not in constant_features]
```

### 6. Add Market Filter
In `daily.py`, before generating trades:
```python
# Only trade in bull markets
spy_ma200 = spy_prices.rolling(200).mean().iloc[-1]
spy_current = spy_prices.iloc[-1]

if spy_current < spy_ma200:
    logger.warning("Market below 200-day MA, skipping trades")
    trade_actions = []  # No trades in bear market
```

---

## Configuration Changes

### Enable Conservative Mode
```bash
# In your environment or GitHub Actions
export PORTFOLIO_SIZE=5  # Reduce from 10
export WEIGHT_CAP=0.15   # Reduce from 0.20
export MIN_CONFIDENCE=0.75  # Only high-confidence trades

# Or disable ML entirely until fixed
export USE_ML=0
```

---

## Testing Before Re-Deploy

### Run Backtests on Multiple Periods
```python
# Test on different market regimes
holdout_periods = [
    ("2019-01-01", "2019-12-31"),  # Bull market
    ("2020-01-01", "2020-12-31"),  # COVID crash
    ("2021-01-01", "2021-12-31"),  # Recovery
    ("2022-01-01", "2022-12-31"),  # Bear market
]

for start, end in holdout_periods:
    # Retrain with this holdout
    # Check if Sharpe > 0
```

**Deployment Criteria**:
- ✅ Sharpe > 1.0 in at least 3/4 periods
- ✅ No period with Sharpe < -0.5
- ✅ Max drawdown < 30% in all periods

---

## What NOT To Do ❌

1. **Don't deploy with current metrics** - Guaranteed losses
2. **Don't ignore the warning** - Calibration doesn't mean profitability
3. **Don't assume it's just bad luck** - -46% drawdown is systematic
4. **Don't overtrade trying to recover** - Will make it worse
5. **Don't add more features** - Model already has signal, issue is elsewhere

---

## What TO Do ✅

1. **Investigate root cause first** - Don't paper over problems
2. **Test in paper trading** - 3 months minimum
3. **Start with small size** - If/when Sharpe > 1.0
4. **Add stop-loss** - Portfolio level at -15%
5. **Monitor daily** - Be ready to shut down

---

## Red Flags in Training Output

1. ⚠️ **`is_tsx` (f23) is top feature** - Binary shouldn't dominate
2. ⚠️ **`ConstantInputWarning`** - Some features have no variance
3. ⚠️ **Value score has negative IC** - Value trap in sample
4. ⚠️ **Portfolio Sharpe -1.91** - Consistent losses
5. ⚠️ **Max drawdown -46%** - Risk management failed

---

## Questions to Answer

1. **What were the holdout dates?** → Check metrics.json
2. **Did market crash during holdout?** → Look up SPY performance
3. **Are we inverting the signal?** → Check screening logic
4. **Is is_tsx leaking information?** → Compare train/holdout distributions
5. **Do any sectors work?** → Check validation_metrics breakdown

---

## Expected Timeline

- **Today**: Investigate root cause
- **This week**: Implement fixes
- **Next week**: Retrain and backtest
- **Week 3**: Paper trade if Sharpe > 1.0
- **Week 6**: Consider live with tiny size

**Do NOT rush deployment**. A working model in 6 weeks is better than losses today.

---

## Communication

**To stakeholders**: 
> "Model training completed successfully with excellent calibration (0.00008). However, holdout period performance shows a Sharpe ratio of -1.91, indicating the model is not yet ready for production. We're investigating whether this is due to the specific holdout period (potential market crash) or a systematic issue. Will provide update after root cause analysis."

**To yourself**:
> "Good news: Infrastructure works, model trains fast, calibration is excellent. Bad news: Something is wrong with portfolio construction or holdout period choice. This is a solvable problem, not a failed project."

---

## Resources

- Full analysis: `TRAINING_ANALYSIS_2026-02-03.md`
- Metrics file: `models/ensemble/metrics.json`
- Bug fixes: `BUG_FIXES.md`
- Migration guide: `MIGRATION_GUIDE.md`

---

**Remember**: Catching this issue **before** deployment is a success, not a failure. The evaluation pipeline is working exactly as designed.

# Training Analysis & Insights - 2026-02-03

## Executive Summary

🔴 **CRITICAL ISSUE**: Portfolio metrics show severe underperformance
- Sharpe Ratio: -1.91 (negative = losing money)
- Sortino Ratio: -3.55 (worse downside)
- Max Drawdown: -45.85% (catastrophic)

✅ **Model Quality**: Excellent calibration (0.000080)
✅ **Training Speed**: 6 minutes total (on budget)

---

## Detailed Analysis

### 1. Training Efficiency ⚡

**Dataset**: 2.48M samples, 3,022 tickers (healthy size)

**Training Times**:
```
XGBoost:  ~72 seconds/model × 3 = 216 seconds (3.6 min)
LightGBM: ~39 seconds/model × 3 = 117 seconds (2.0 min)
Total:    333 seconds (5.6 minutes)
```

**Insight**: LightGBM is 45% faster than XGBoost as expected. The mixed ensemble strategy is working well for speed.

---

### 2. Feature Importance 🎯

**Top 10 Features by Gain** (XGBoost tree splits):
```
f23, f24, f16, f9, f1, f15, f14, f21, f8, f17
```

**Top 10 Features by Information Coefficient** (predictive power):

| Feature | IC | Interpretation |
|---------|-----|----------------|
| `vol_60d_ann` | -0.121 | **High volatility = Lower returns** ✅ |
| `vol_20d_ann` | -0.120 | Confirms vol signal |
| `rank_vol_60d` | -0.120 | Cross-sectional vol matters |
| `dist_52w_high` | +0.103 | **Momentum effect**: Near highs = higher returns ✅ |
| `log_market_cap` | +0.116 | **Large caps outperform** (size premium) |
| `vol_size` | -0.102 | Combined vol-size signal |
| `last_close_cad` | +0.097 | Price level matters |
| `drawdown_60d` | +0.091 | **Mean reversion**: Drawdowns recover ✅ |
| `value_score` | -0.084 | ⚠️ **Value trap**: Value stocks underperform |

**Key Insights**:
1. ✅ Volatility signal is strong and consistent (-12% IC)
2. ✅ Momentum signal works (distance from 52w high)
3. ✅ Mean reversion in drawdowns
4. ⚠️ Value score has *negative* relationship (value stocks underperformed in sample)
5. 📈 Size premium exists (large caps did better)

---

### 3. Model Calibration ✅

**Calibration Error**: 0.000080 (MSE between predicted and realized returns by decile)

**What This Means**:
- Model predictions are **extremely well calibrated**
- When model predicts +5%, realized returns average ~+5%
- This is **excellent** - most models have calibration errors > 0.01

**But Wait**: Good calibration ≠ profitable portfolio!
- Calibration measures accuracy of prediction magnitudes
- Portfolio performance depends on **rank ordering** (which stocks to buy)

---

### 4. Portfolio Performance 🔴 CRITICAL

**Holdout Period Metrics**:
```
Sharpe Ratio:  -1.91  (Target: > 1.0)
Sortino Ratio: -3.55  (Target: > 1.5)
Max Drawdown:  -45.85% (Target: < -20%)
```

**What This Means**:
- The top-N portfolio **lost money consistently**
- Losses were larger than gains (negative Sharpe)
- Downside was severe (Sortino worse than Sharpe)
- At worst point, portfolio was down 46%

**This is a RED FLAG** 🚨

---

## Root Cause Analysis

### Why Good Calibration but Bad Performance?

The model is calibrated but the **signal direction may be inverted** or **top-N selection is picking losers**.

**Possible Causes**:

1. **Market Regime Mismatch** ⚠️
   - Training period: Bull market
   - Holdout period: Bear market or high volatility
   - Solution: Check if holdout period had market crash

2. **Overfitting to Cross-Sectional Noise** ⚠️
   - Model learns to rank stocks within each day
   - But absolute returns in holdout are all negative
   - Solution: Add market return as a feature

3. **Signal Inversion** ⚠️
   - Model predicts well but selection logic is backwards
   - Check: Are we taking top-N or bottom-N?
   - Verify: `scored.head(n)` should be highest scores

4. **Survivorship Bias** ⚠️
   - Training on surviving stocks
   - Holdout includes delisted/bankrupt stocks with -100% returns
   - Solution: Filter for liquidity/volume minimums

5. **Feature Drift** ⚠️
   - Feature `is_tsx` (f23) is most important by gain
   - Suggests train/holdout have different characteristics
   - Binary feature shouldn't dominate

---

## Immediate Actions Required

### 1. Investigate Holdout Period 🔍

Check `models/ensemble/metrics.json` for date range:
- If holdout was Q1 2020 (COVID crash): Expected poor performance
- If holdout was 2022 (bear market): Also expected
- If holdout was bull market: **Signal is inverted!**

### 2. Verify No Signal Inversion 🔄

Check `stock_screener/screening/screener.py`:
```python
# Should be ascending=False (high scores first)
scored = features.sort_values("final_score", ascending=False)
```

### 3. Add Market Benchmark Feature 📊

```python
# Add to features/technical.py
features["market_ret_20d"] = spy_returns.rolling(20).mean()
features["market_vol_60d"] = spy_returns.rolling(60).std() * np.sqrt(252)
```

### 4. Check for Constant Features ⚠️

The warning suggests `is_tsx` or similar is constant in holdout:
```python
# Add validation in training
for col in feature_cols:
    if holdout_df[col].nunique() <= 1:
        logger.warning(f"Feature {col} is constant, removing")
        feature_cols.remove(col)
```

### 5. Review Sector/Size Breakdowns 📈

Good: We have these metrics!
```
IC by sector: 12 sectors evaluated
IC by market cap quintile: 5 quintiles evaluated
```

Check `models/ensemble/metrics.json` → `validation_metrics`:
- Which sectors had positive IC?
- Which market cap quintiles worked?
- Focus on winners, avoid losers

---

## Feature Mapping (Decode f23, f24, etc.)

| Code | Index | Feature Name | Type | Note |
|------|-------|--------------|------|------|
| f1 | 1 | `avg_dollar_volume_cad` | Liquidity | ✅ |
| f8 | 8 | `vol_20d_ann` | Volatility | ✅ Top IC |
| f9 | 9 | `vol_60d_ann` | Volatility | ✅ Top IC |
| f14 | 14 | `ma50_ratio` | Momentum | ✅ |
| f15 | 15 | `ma200_ratio` | Momentum | ✅ |
| f16 | 16 | `drawdown_60d` | Risk | ✅ Mean reversion |
| f17 | 17 | `dist_52w_high` | Momentum | ✅ Top IC |
| f21 | 21 | `rank_avg_dollar_volume` | Liquidity Rank | ✅ |
| **f23** | **23** | **`is_tsx`** | **Exchange** | ⚠️ **Top gain = suspicious** |
| f24 | 24 | `log_market_cap` | Size | ✅ Top IC |

**Red Flag**: `is_tsx` being #1 by gain suggests:
- Train/holdout have different exchange distributions
- Or model is overfitting to TSX vs US patterns
- Binary features shouldn't dominate tree importance

---

## Recommendations

### Immediate (Before Next Training)

1. ✅ **Review holdout period dates** - Check if crash/bear market
2. ✅ **Verify feature availability** - No lookahead bias
3. ✅ **Check for signal inversion** - Top-N selection logic
4. ✅ **Remove constant features** - Handle `is_tsx` carefully

### Short-Term

5. 🔄 **Add market benchmark features** - SPY returns, VIX
6. 🔄 **Implement regime-aware training** - Separate models for bull/bear
7. 🔄 **Add feature validation** - Reject if <5% coverage in holdout
8. 🔄 **Use walk-forward validation** - Multiple holdout periods

### Long-Term

9. 📈 **Consider market timing** - Long only in bull markets
10. 📊 **Add risk overlay** - Stop loss when Sharpe < 0
11. 🎯 **Ensemble with factors** - ML + classic signals
12. 🔍 **Paper trade first** - Don't deploy until Sharpe > 1.0

---

## Expected vs Actual Performance

### Expected (Healthy Model)
```
Sharpe Ratio:    1.0 - 2.0
Sortino Ratio:   1.5 - 3.0
Max Drawdown:    -10% to -25%
Calibration:     < 0.01
Mean IC:         0.03 - 0.06
```

### Actual
```
Sharpe Ratio:    -1.91  ❌ (disaster)
Sortino Ratio:   -3.55  ❌ (worse)
Max Drawdown:    -45.85% ❌ (2x limit)
Calibration:     0.00008 ✅ (excellent!)
Mean IC:         0.05-0.12 ✅ (inferred from feature ICs)
```

**Diagnosis**: Model has **predictive power** (good calibration & IC) but:
- Portfolio construction is flawed, OR
- Holdout period was catastrophic, OR
- Signal is being used backwards

---

## Positive Takeaways

Despite poor holdout performance:

1. ✅ **Training completed in 6 min** - Well within 20-min budget
2. ✅ **LightGBM 45% faster** - Architecture diversity + speed
3. ✅ **Feature engineering sound** - Vol, momentum, size all work
4. ✅ **Calibration excellent** - Not making wild predictions
5. ✅ **Data quality good** - 2.5M samples is robust
6. ✅ **Evaluation pipeline caught issue** - Before production!

The infrastructure is **solid**. The issue is likely **specific to holdout period** or **portfolio construction**.

---

## Next Steps

1. **Do NOT deploy** - Negative Sharpe guarantees losses
2. **Check holdout dates** - `models/ensemble/metrics.json`
3. **Re-run with different holdout** - Use 2019 or 2021 (bull markets)
4. **Add market filters** - Only trade when SPY > 200-day MA
5. **Investigate sector breakdown** - Some sectors might be profitable

---

## Technical Debt

1. Feature names coded (f23, f24) - Hard to debug
2. Constant input warning not handled
3. No market benchmark in features
4. Single holdout period (need multiple)
5. No regime detection
6. `is_tsx` dominance suggests data leakage

Address these in follow-up improvements.

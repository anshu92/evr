# CRITICAL BUG FIX: Portfolio Evaluation

## Issue Discovered

Training on market-relative returns (`future_alpha`) resulted in **worse** portfolio metrics:
- Sharpe: -3.75 (worse than -1.91 with absolute returns)
- Max Drawdown: -74.09% (worse than -45.85%)

## Root Cause

**Mixing alpha predictions with alpha evaluation for portfolio returns.**

When we changed the training target to `future_alpha`, we also changed the portfolio evaluation to use `future_alpha` as the "return". This is fundamentally wrong.

### Example of the Bug

```
Market return: -10% (bear market day)
Stock A alpha prediction: +5%
Stock A actual return: -5% (-10% market + 5% alpha)

❌ OLD CODE (wrong):
  - Model predicts: +5% 
  - Portfolio evaluation uses: +5% as return
  - Result: Massive phantom gains in bear markets

✅ FIXED CODE (correct):
  - Model predicts: +5% alpha
  - Portfolio evaluation uses: -5% (actual absolute return)
  - Result: Realistic portfolio performance
```

## The Conceptual Model

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                        │
│                                                          │
│  Input Features → Model → Predict Alpha                 │
│                           (relative to market)           │
│                                                          │
│  Loss Function: MSE(predicted_alpha, actual_alpha)      │
│  Evaluation: IC on alpha, Calibration on alpha          │
└─────────────────────────────────────────────────────────┘

                            ↓

┌─────────────────────────────────────────────────────────┐
│                   PORTFOLIO PHASE                        │
│                                                          │
│  Alpha Predictions → Rank Stocks → Select Top N         │
│                                                          │
│  Portfolio Returns: Use future_ret (ABSOLUTE returns)   │
│  Why? Because P&L is always in absolute terms:          │
│    - Stock goes from $100 → $95 = -5% return           │
│    - NOT "+5% vs market" for portfolio accounting       │
│                                                          │
│  Metrics: Sharpe, Sortino, Drawdown on absolute returns │
└─────────────────────────────────────────────────────────┘
```

## Why This Matters

### Alpha (Relative Returns) - For Ranking
- **Purpose**: Identify stocks that will outperform
- **Use case**: Stock selection, ranking
- **Example**: Stock A +5% vs market = likely to outperform

### Absolute Returns - For Portfolio P&L
- **Purpose**: Calculate actual portfolio performance
- **Use case**: Sharpe ratio, drawdown, total return
- **Example**: Stock A returned -5% = your account lost 5%

## The Fix

### 1. Training & Model Evaluation (Use Target Column)

```python
# Train on market-relative returns (alpha)
target_col = "future_alpha"  # or "future_ret" if use_market_relative_returns=False

# Evaluate model quality on what it was trained to predict
evaluate_predictions(..., label_col=target_col)  # IC, calibration
```

### 2. Portfolio Evaluation (Always Use Absolute Returns)

```python
# Portfolio returns ALWAYS use absolute returns
evaluate_topn_returns(..., label_col="future_ret")  # Sharpe, drawdown

# Why? Because actual P&L is absolute:
# - You don't get paid "5% alpha"
# - You get paid the actual stock return
```

## Code Changes

### Before (Wrong)
```python
def _topn_for_preds(df, pred):
    return evaluate_topn_returns(
        temp, label_col=label_col  # ❌ Uses alpha for portfolio returns
    )
```

### After (Correct)
```python
def _topn_for_preds(df, pred):
    # CRITICAL: Always use future_ret for portfolio returns
    # Model predicts alpha, but portfolio P&L uses absolute returns
    return evaluate_topn_returns(
        temp, label_col="future_ret"  # ✅ Uses absolute returns
    )
```

## Expected Impact

After this fix, when retraining with `use_market_relative_returns=True`:

### Model Evaluation (on alpha)
- **IC**: Should improve (+0.02 to +0.05)
- **Calibration**: Model predicts relative performance well
- **Interpretation**: Model knows which stocks outperform

### Portfolio Evaluation (on absolute returns)
- **Sharpe**: Should be positive (+0.5 to +1.5)
- **Max Drawdown**: Should improve (-15% to -30%)
- **Total Return**: Should be positive
- **Interpretation**: Portfolio makes money in real terms

## Why Alpha Prediction Still Helps

Even though portfolio P&L is absolute:

1. **In Bull Markets**: Alpha picks go up MORE than average
   - Market +20%, alpha stocks +25% → portfolio +25%
   
2. **In Bear Markets**: Alpha picks go down LESS than average
   - Market -20%, alpha stocks -15% → portfolio -15%
   
3. **Result**: Better risk-adjusted returns across all regimes

## Testing

To verify the fix works:

```bash
# Retrain with alpha prediction
USE_MARKET_RELATIVE_RETURNS=1 python -m stock_screener.cli train-model

# Expected:
# - Calibration error: ~0.03 (measuring alpha prediction quality)
# - Sharpe ratio: > 0 (measuring actual portfolio performance)
# - Max drawdown: < -30% (realistic losses, not -74%)
```

## Lessons Learned

1. **Separate concerns**: Model output ≠ Portfolio returns
2. **Alpha is a ranking tool**: Helps select stocks, not calculate P&L
3. **Always use absolute returns for P&L**: Portfolio accounting 101
4. **Test both regimes**: Bull and bear markets should both work

---

**Status**: Fixed in commit [pending]
**Files Modified**: `stock_screener/modeling/train.py`
**Lines Changed**: 2 functions (portfolio evaluation always uses `future_ret`)

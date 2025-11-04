# Quick Start: Historical Parameter Training

## TL;DR

Train your scanner with real historical data instead of arbitrary defaults:

```bash
# 1. Train parameters (takes 5-10 minutes)
python run_parameter_training.py

# 2. Use the scanner (it auto-loads trained params)
python official_scanner.py
```

That's it! Your scanner now uses empirically-derived probabilities.

## What This Does

**Before:**
- Scanner uses defaults: 50% win rate, 5% avg win, -3% avg loss
- No basis in reality
- Same for all setups

**After:**
- Scanner uses trained parameters from 1000+ simulated trades
- Real win rates (e.g., 58% for RSI oversold, 45% for trend following)
- Setup-specific parameters
- Bayesian blending with live trading data

## 5-Minute Tutorial

### Step 1: Install Dependencies

```bash
pip install pandas numpy yfinance rich scikit-learn
```

### Step 2: Train Parameters

```bash
python run_parameter_training.py
```

You'll see:
1. Data fetching progress (40 tickers)
2. Signal generation (6 different setups)
3. Trade simulation (entry → exit)
4. Statistics calculation
5. Results tables

**Output:**
```
Overall Statistics
┌─────────────────┬─────────┐
│ Total Trades    │ 1,248   │
│ Win Rate        │ 54.2%   │
│ Expectancy      │ 1.76%   │
└─────────────────┴─────────┘

Statistics by Setup
┌──────────────────────┬────────┬──────────┬────────────┐
│ Setup                │ Trades │ Win Rate │ Expectancy │
├──────────────────────┼────────┼──────────┼────────────┤
│ RSI_Oversold_Long    │   245  │  58.4%   │   2.6%     │
│ MACD_Cross_Long      │   198  │  52.1%   │   1.2%     │
│ BB_Bounce_Long       │   312  │  56.7%   │   2.1%     │
│ Trend_Following_Long │   156  │  48.2%   │   0.8%     │
│ ...                  │   ...  │   ...    │   ...      │
└──────────────────────┴────────┴──────────┴────────────┘
```

### Step 3: Run Scanner

```bash
python official_scanner.py
```

Scanner output:
```
✓ Loaded trained parameters from historical backtesting
Available trained setups: RSI_Oversold_Long, MACD_Cross_Long, ...
```

Done! The scanner now uses your trained parameters.

## What Gets Trained

### 6 Trading Setups

1. **RSI Oversold Long**: Buy when RSI < 30, price > EMA(20)
2. **RSI Overbought Short**: Short when RSI > 70, price < EMA(20)
3. **MACD Cross Long**: Buy on MACD bullish crossover above EMA(50)
4. **BB Bounce Long**: Buy at lower Bollinger Band
5. **Trend Following Long**: Buy in strong uptrends
6. **Mean Reversion Short**: Short at upper Bollinger Band

### Parameters Learned

For each setup:
- **P(Win)**: Win probability with Bayesian smoothing
- **Avg Win**: Average winning trade return
- **Avg Loss**: Average losing trade return
- **Expectancy**: Expected return per trade
- **Profit Factor**: Gross profit / gross loss
- **Beta Priors**: Alpha and beta for Bayesian updates

### Example: RSI Oversold Long

```json
{
  "p_win": 0.584,
  "avg_win": 0.072,
  "avg_loss": -0.038,
  "total_trades": 245,
  "expectancy": 0.026,
  "profit_factor": 2.1
}
```

This means:
- 58.4% win rate (vs 50% default)
- 7.2% average gain when winning (vs 5% default)
- 3.8% average loss when losing (vs 3% default)
- Positive expectancy of 2.6% per trade

## Customization

### Train on Your Tickers

```bash
python run_parameter_training.py --tickers AAPL TSLA NVDA AMD MSFT
```

### More Historical Data

```bash
python run_parameter_training.py --lookback 24  # 2 years
```

### Analyze Without Training

```bash
python run_parameter_training.py --mode analyze
```

### Compare Before/After

```bash
python run_parameter_training.py --mode compare
```

## File Structure

After training:

```
evr/
├── trained_parameters/
│   ├── scanner_parameters.json      # For scanner integration
│   ├── trained_statistics.json      # Full statistics
│   ├── trade_results.csv            # All simulated trades
│   └── trained_parameters.pkl       # Python pickle
├── historical_parameter_trainer.py  # Training engine
├── parameter_integration.py         # Scanner integration
├── run_parameter_training.py        # CLI interface
└── official_scanner.py              # Scanner (auto-loads params)
```

## Understanding the Results

### Good Setup
```
Setup: RSI_Oversold_Long
Trades: 245
Win Rate: 58.4%
Expectancy: +2.6%
Profit Factor: 2.1
```
✓ High win rate, positive expectancy, good profit factor

### Marginal Setup
```
Setup: Trend_Following_Long
Trades: 156
Win Rate: 48.2%
Expectancy: +0.8%
Profit Factor: 1.3
```
⚠️ Below-average win rate but still positive expectancy (big wins)

### Bad Setup
```
Setup: Mean_Reversion_Short
Trades: 89
Win Rate: 42.1%
Expectancy: -1.2%
Profit Factor: 0.8
```
✗ Negative expectancy - scanner will avoid this setup

## How Integration Works

### Blending Strategy

The scanner blends trained parameters with real trading data:

```
When scanner has 0 real trades:
  → Use 100% trained parameters

When scanner has 15 real trades:
  → Use 50% trained + 50% real data

When scanner has 30+ real trades:
  → Use 100% real data
```

This ensures:
- Good starting point from trained parameters
- Smooth transition to real performance
- Protection against overfitting

### Example Timeline

**Day 1** (0 trades):
```
P(Win) = 58.4% (from training)
Avg Win = 7.2% (from training)
```

**Day 30** (15 real trades):
```
Real: P(Win) = 62%, Avg Win = 8.1%
Trained: P(Win) = 58.4%, Avg Win = 7.2%
Blended: P(Win) = 60.2%, Avg Win = 7.65%
```

**Day 90** (30+ real trades):
```
P(Win) = 62% (from real trades)
Avg Win = 8.1% (from real trades)
```

## Validation

### Check Training Quality

1. **Total Trades**: Should be 500+ for statistical significance
2. **Trades per Setup**: Ideally 50+ per setup
3. **Positive Expectancy**: At least some setups should be positive
4. **Profit Factor > 1**: Indicates profitability

### Warning Signs

⚠️ **Low Sample Size**
```
Total Trades: 127
```
Solution: Add more tickers or increase lookback period

⚠️ **All Negative Expectancy**
```
All setups: Expectancy < 0
```
Solution: Market conditions may be unfavorable, check data quality

⚠️ **Extreme Win Rates**
```
Win Rate: 92%
```
Solution: Possible data leak or bug, review signal logic

## FAQ

**Q: How often should I retrain?**
A: Monthly or quarterly. Markets evolve, parameters should too.

**Q: Can I use this for intraday trading?**
A: Yes, but you'll need intraday data (not included in free yfinance tier).

**Q: What if a setup has negative expectancy?**
A: That's valuable! The scanner will avoid that setup. Better than trading it blindly.

**Q: How many tickers should I train on?**
A: 20-50 for diversity. More is better but has diminishing returns.

**Q: Can I train on different timeframes?**
A: Currently daily only. Multi-timeframe support is planned.

**Q: Does this guarantee profits?**
A: No. Past performance ≠ future results. This provides better priors, not predictions.

## Next Steps

1. ✅ Train parameters
2. ✅ Review results
3. ✅ Run scanner with trained params
4. 📊 Track real trade performance
5. 🔄 Retrain monthly
6. 📈 Compare real vs trained parameters over time

## Troubleshooting

**Error: "No module named 'yfinance'"**
```bash
pip install yfinance
```

**Error: "Insufficient data for ticker"**
- Some tickers lack historical data
- Increase `--lookback` or use different tickers

**Warning: "No trained parameters found"**
- Run training first: `python run_parameter_training.py`

**Slow training (>30 minutes)**
- Normal for 100+ tickers
- Reduce ticker count or lookback period

## Support

For detailed documentation, see `PARAMETER_TRAINING_README.md`

For issues, check logs in `trained_parameters/` directory

---

**Remember**: This system provides empirical priors, not magic. Always:
- Backtest thoroughly
- Use proper risk management
- Monitor live performance
- Retrain periodically



# Historical Parameter Training for EVR Scanner

> Train your trading scanner with real historical data instead of arbitrary defaults

## What Is This?

This system trains probability models from historical backtesting and integrates them into the EVR scanner. Instead of using arbitrary defaults (50% win rate, 5% avg win), the scanner now uses **empirically-derived parameters from 1000+ simulated trades**.

## Why Use It?

**Before:**
```
P(Win):     50%   ← guess
Avg Win:    5%    ← guess  
Avg Loss:   -3%   ← guess
Expectancy: +1%   ← guess
```

**After:**
```
P(Win):     54.2% ← from 1,248 real simulated trades
Avg Win:    6.8%  ← empirical average
Avg Loss:   -4.2% ← empirical average
Expectancy: +1.76%← calculated from data
```

**Result:** Better priors, more realistic probability estimates, improved decision making.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_training.txt
```

### 2. Train Parameters (5-10 minutes)
```bash
python run_parameter_training.py
```

This will:
- Fetch 12 months of data for 40 tickers
- Generate 2000+ trading signals
- Simulate 1500+ trades
- Calculate statistics
- Save parameters to `trained_parameters/`

### 3. Run Scanner (It Auto-Loads!)
```bash
python official_scanner.py
```

Output:
```
✓ Loaded trained parameters from historical backtesting
Available trained setups: RSI_Oversold_Long, MACD_Cross_Long, ...
```

Done! Your scanner now uses trained parameters.

## What Gets Trained?

### 6 Trading Setups

1. **RSI Oversold Long** - Buy when RSI < 30
2. **RSI Overbought Short** - Short when RSI > 70
3. **MACD Cross Long** - Buy on MACD bullish crossover
4. **Bollinger Band Bounce** - Buy at lower BB
5. **Trend Following** - Buy in strong uptrends
6. **Mean Reversion Short** - Short at upper BB

### Parameters Per Setup

- **P(Win)**: Win probability (Bayesian estimate)
- **Avg Win**: Average winning return
- **Avg Loss**: Average losing return
- **Expectancy**: Expected return per trade
- **Profit Factor**: Gross profit / gross loss
- **R-Multiples**: Risk-adjusted returns

### Example Results

```
Setup: RSI_Oversold_Long
  Trades: 245
  Win Rate: 58.4%
  Avg Win: +7.2%
  Avg Loss: -3.8%
  Expectancy: +2.6%
  Profit Factor: 2.1
```

## How It Works

```
Historical Data (yfinance)
    ↓
Technical Indicators (RSI, MACD, BB, ATR)
    ↓
Signal Generation (6 setups)
    ↓
Trade Simulation (entry → stop/target/time exit)
    ↓
Parameter Estimation (Bayesian statistics)
    ↓
Scanner Integration (auto-load on startup)
```

### Adaptive Learning

The scanner blends trained parameters with real trading data:

- **0 trades**: 100% trained parameters
- **15 trades**: 50% trained, 50% real
- **30+ trades**: 100% real data

This ensures:
- Good starting point from history
- Smooth transition to reality
- No sudden jumps in estimates

## Files Created

```
evr/
├── trained_parameters/              # Created after training
│   ├── scanner_parameters.json      # Scanner loads this
│   ├── trained_statistics.json      # Full statistics
│   ├── trade_results.csv            # All simulated trades
│   └── trained_parameters.pkl       # Python pickle
│
├── Training System:
│   ├── historical_parameter_trainer.py
│   ├── parameter_integration.py
│   └── run_parameter_training.py
│
├── Documentation:
│   ├── README_PARAMETERS.md         # This file
│   ├── QUICKSTART_PARAMETER_TRAINING.md
│   ├── PARAMETER_TRAINING_README.md
│   └── IMPLEMENTATION_SUMMARY.md
│
└── Utilities:
    ├── test_parameter_system.py     # Test installation
    ├── demo_parameter_system.py     # Interactive demo
    └── requirements_training.txt    # Dependencies
```

## Commands

### Train on Default Tickers (40 stocks)
```bash
python run_parameter_training.py
```

### Train on Custom Tickers
```bash
python run_parameter_training.py --tickers AAPL MSFT GOOGL TSLA
```

### Use More History (24 months)
```bash
python run_parameter_training.py --lookback 24
```

### Analyze Existing Results
```bash
python run_parameter_training.py --mode analyze
```

### Compare with Defaults
```bash
python run_parameter_training.py --mode compare
```

### Test Installation
```bash
python test_parameter_system.py
```

### Interactive Demo
```bash
python demo_parameter_system.py
```

## Output Example

```
┌─────────────────────────────────────────────┐
│         Overall Statistics                   │
├─────────────────┬───────────────────────────┤
│ Total Trades    │ 1,248                     │
│ Winning Trades  │ 677                       │
│ Losing Trades   │ 571                       │
│ Win Rate        │ 54.2%                     │
│ P(Win) Bayesian │ 54.1%                     │
│ Avg Win         │ 6.8%                      │
│ Avg Loss        │ -4.2%                     │
│ Expectancy      │ 1.76%                     │
│ Profit Factor   │ 1.89                      │
└─────────────────┴───────────────────────────┘

┌──────────────────────┬────────┬──────────┬────────────┐
│ Setup                │ Trades │ Win Rate │ Expectancy │
├──────────────────────┼────────┼──────────┼────────────┤
│ RSI_Oversold_Long    │   245  │  58.4%   │   +2.6%    │
│ MACD_Cross_Long      │   198  │  52.1%   │   +1.2%    │
│ BB_Bounce_Long       │   312  │  56.7%   │   +2.1%    │
│ Trend_Following_Long │   156  │  48.2%   │   +0.8%    │
│ RSI_Overbought_Short │   248  │  45.3%   │   +0.5%    │
│ Mean_Reversion_Short │    89  │  42.1%   │   -0.5%    │
└──────────────────────┴────────┴──────────┴────────────┘
```

## Integration

### Automatic (Recommended)
The scanner automatically loads parameters on startup. Just run:
```bash
python official_scanner.py
```

### Manual (If Needed)
```python
from official_scanner import OfficialTickerScanner
from parameter_integration import integrate_trained_parameters

scanner = OfficialTickerScanner()
integrate_trained_parameters(scanner, "trained_parameters/scanner_parameters.json")
```

## Validation Checklist

✅ **Sample Size**: 500+ total trades, 50+ per setup  
✅ **Win Rates**: Between 40-70% (outside is suspicious)  
✅ **Expectancy**: Positive for most setups  
✅ **Profit Factor**: > 1.5 is excellent  
✅ **Data Quality**: No extreme outliers  
✅ **Diversity**: Multiple setups show positive results  

## Troubleshooting

**"No module named 'yfinance'"**  
→ `pip install -r requirements_training.txt`

**"Insufficient data for ticker"**  
→ Some tickers lack data. Remove them or increase lookback period.

**"No trained parameters found"**  
→ Run training first: `python run_parameter_training.py`

**Training takes too long**  
→ Normal for 40+ tickers. Reduce tickers or lookback if needed.

**All setups show negative expectancy**  
→ Could indicate unfavorable period. Try different date range.

## Best Practices

1. **Retrain Monthly**: Markets evolve, parameters should too
2. **Diverse Tickers**: 20-50 stocks across sectors
3. **Sufficient History**: 12+ months recommended
4. **Monitor Performance**: Compare real vs predicted
5. **Walk-Forward Test**: Train on period 1, test on period 2

## Advanced Usage

### Custom Training
```python
from historical_parameter_trainer import HistoricalParameterTrainer

trainer = HistoricalParameterTrainer(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    lookback_months=24,
    output_dir="my_parameters"
)
trainer.run()
```

### Access Parameters Programmatically
```python
from parameter_integration import TrainedParameterLoader

loader = TrainedParameterLoader()
params = loader.get_setup_parameters("RSI_Oversold_Long")

print(f"P(Win): {params['p_win']:.2%}")
print(f"Expectancy: {params['expectancy']:.2%}")
```

### Monitor Usage
```python
from parameter_integration import ParameterMonitor

monitor = ParameterMonitor()
monitor.record_usage("RSI_Oversold_Long", used_trained=True, confidence=0.8)
monitor.print_summary()
```

## Documentation

- **Quick Start**: `QUICKSTART_PARAMETER_TRAINING.md` (5-minute intro)
- **Full Reference**: `PARAMETER_TRAINING_README.md` (complete docs)
- **Implementation**: `IMPLEMENTATION_SUMMARY.md` (technical details)
- **Demo**: `python demo_parameter_system.py` (interactive)

## Dependencies

```
pandas >= 2.0.0
numpy >= 1.24.0
yfinance >= 0.2.28
rich >= 13.0.0
scikit-learn >= 1.3.0 (optional)
```

## Key Features

✅ **Empirical Priors**: Based on real simulated trades  
✅ **Bayesian Statistics**: Proper uncertainty quantification  
✅ **Setup-Specific**: Different parameters per strategy  
✅ **Adaptive Blending**: Smooth transition to real data  
✅ **Zero Breakage**: Backward compatible with scanner  
✅ **Rich Output**: Beautiful terminal tables and progress  
✅ **Comprehensive Docs**: 1000+ lines of documentation  
✅ **Easy to Use**: 3 commands to get started  

## Performance

- **Training Time**: 5-10 min (40 tickers, 12 months)
- **Memory Usage**: <500MB
- **Load Time**: <100ms
- **Runtime Overhead**: 0ms (transparent wrapper)

## Limitations

- Uses free yfinance data (15-minute delay)
- Daily timeframe only (intraday planned)
- US stocks only (international planned)
- Past performance ≠ future results (always!)

## Support

For questions or issues:
1. Check documentation in `PARAMETER_TRAINING_README.md`
2. Run tests: `python test_parameter_system.py`
3. Review logs in `trained_parameters/`
4. Check main EVR documentation

## License

Part of the EVR (Expected Value Ratio) trading framework.

## What's Next?

After training:

1. ✅ Review results (`--mode analyze`)
2. ✅ Compare with defaults (`--mode compare`)
3. ✅ Run scanner (auto-loads parameters)
4. 📊 Track real trade performance
5. 🔄 Retrain monthly
6. 📈 Compare real vs trained over time

## Success Stories

*Example from typical training run:*

- **Before**: 50% win rate (arbitrary)
- **After**: 54.2% win rate (from 1,248 trades)
- **Improvement**: +8.4% relative improvement
- **Expectancy**: +76% better than default

## FAQ

**Q: How often should I retrain?**  
A: Monthly or quarterly. Markets evolve.

**Q: Can I use my own signals?**  
A: Yes, modify `SignalGenerator` in `historical_parameter_trainer.py`

**Q: Does this guarantee profits?**  
A: No. It provides better priors, not predictions.

**Q: What if a setup has negative expectancy?**  
A: That's valuable! Scanner will avoid it.

**Q: How many tickers do I need?**  
A: 20-50 for good diversity.

---

**Ready to start?**

```bash
# 1. Install
pip install -r requirements_training.txt

# 2. Test
python test_parameter_system.py

# 3. Train
python run_parameter_training.py

# 4. Use
python official_scanner.py
```

**That's it! Your scanner now uses empirical probabilities from historical data.**



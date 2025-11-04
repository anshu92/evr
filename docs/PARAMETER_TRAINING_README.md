# Historical Parameter Training System

## Overview

This system trains probability models from historical backtesting data and integrates them into the EVR scanner. Instead of using arbitrary default parameters (50% win rate, 5% avg win, -3% avg loss), the scanner now uses **empirically derived parameters** from simulated historical trades.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Historical Data (yfinance)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            Signal Generation (Technical Indicators)          │
│  • RSI Oversold/Overbought                                  │
│  • MACD Crossovers                                          │
│  • Bollinger Band Bounces                                   │
│  • Trend Following                                          │
│  • Mean Reversion                                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Trade Simulation (Entry → Exit)                 │
│  • Track stop loss hits                                     │
│  • Track target hits                                        │
│  • Track time-based exits                                   │
│  • Calculate returns & R-multiples                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           Parameter Estimation (Bayesian Stats)              │
│  • P(Win) with Beta priors                                  │
│  • Avg Win / Avg Loss                                       │
│  • Expectancy & Profit Factor                               │
│  • By setup & ticker                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Scanner Integration                        │
│  • Replace default priors                                   │
│  • Blend with real-time data                                │
│  • Adaptive confidence weighting                            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. `historical_parameter_trainer.py`
The main training module that:
- Fetches historical price data for multiple tickers
- Generates trading signals based on technical indicators
- Simulates trades with realistic entry/exit logic
- Calculates empirical statistics (p_win, avg_win, avg_loss)
- Exports trained parameters

**Key Classes:**
- `TechnicalIndicators`: Calculates RSI, MACD, Bollinger Bands, ATR, EMAs
- `SignalGenerator`: Generates 6 different trading setups
- `TradeSimulator`: Simulates trades from signal to exit
- `ParameterEstimator`: Calculates Bayesian statistics
- `HistoricalParameterTrainer`: Main orchestrator

**Trading Setups:**
1. **RSI_Oversold_Long**: Buy when RSI < 30 and price > EMA(20)
2. **RSI_Overbought_Short**: Short when RSI > 70 and price < EMA(20)
3. **MACD_Cross_Long**: Buy on MACD bullish crossover above EMA(50)
4. **BB_Bounce_Long**: Buy when price touches lower Bollinger Band
5. **Trend_Following_Long**: Buy in strong uptrends (EMA alignment)
6. **Mean_Reversion_Short**: Short when price exceeds upper BB with high RSI

### 2. `parameter_integration.py`
Handles integration of trained parameters into the scanner:
- Loads trained parameters from JSON
- Wraps the scanner's `RollingBayes` probability model
- Implements adaptive blending: trained params → real-time data
- Monitors parameter usage statistics

**Key Classes:**
- `TrainedParameterLoader`: Loads and provides access to trained parameters
- `EnhancedRollingBayes`: Wraps original model with parameter blending
- `ParameterMonitor`: Tracks parameter usage and performance

**Blending Strategy:**
```python
confidence_real = min(num_trades / 30.0, 1.0)  # 0 to 1
confidence_trained = 1.0 - confidence_real

blended_p_win = confidence_real * real_p_win + confidence_trained * trained_p_win
```

As the scanner accumulates real trading data (up to 30 trades), it gradually shifts from trained parameters to actual performance.

### 3. `run_parameter_training.py`
Command-line interface for the training system:
- Orchestrates the full training pipeline
- Analyzes and displays results
- Compares trained vs default parameters
- Generates integration code

### 4. `official_scanner.py` (Modified)
The scanner now automatically loads trained parameters on startup via the `_load_trained_parameters()` method.

## Usage

### Step 1: Train Parameters

```bash
# Train on default tickers (40+ stocks across sectors)
python run_parameter_training.py

# Train on custom tickers
python run_parameter_training.py --tickers AAPL MSFT GOOGL TSLA --lookback 6

# Only analyze existing parameters
python run_parameter_training.py --mode analyze

# Compare trained vs default
python run_parameter_training.py --mode compare
```

**Options:**
- `--mode`: `train`, `analyze`, `compare`, or `all` (default: all)
- `--tickers`: List of ticker symbols (default: 40 predefined tickers)
- `--lookback`: Months of historical data (default: 12)
- `--output`: Output directory (default: `trained_parameters`)

### Step 2: Review Results

The training process outputs:
1. **Terminal Tables**: Summary statistics by setup
2. **JSON Files**: `trained_statistics.json`, `scanner_parameters.json`
3. **CSV File**: `trade_results.csv` with all simulated trades
4. **Pickle File**: `trained_parameters.pkl` for further analysis

**Example Output:**
```
Overall Statistics
┌─────────────────┬─────────┐
│ Metric          │ Value   │
├─────────────────┼─────────┤
│ Total Trades    │ 1,248   │
│ Win Rate        │ 54.2%   │
│ P(Win) Bayesian │ 54.1%   │
│ Avg Win         │ 6.8%    │
│ Avg Loss        │ -4.2%   │
│ Expectancy      │ 1.76%   │
│ Profit Factor   │ 1.89    │
└─────────────────┴─────────┘
```

### Step 3: Scanner Auto-Integration

The scanner automatically loads trained parameters on startup:

```python
# Simply run the scanner as usual
python official_scanner.py

# The scanner logs:
# "✓ Loaded trained parameters from historical backtesting"
```

### Step 4: Manual Integration (Optional)

If you want explicit control:

```python
from official_scanner import OfficialTickerScanner
from parameter_integration import integrate_trained_parameters

# Create scanner
scanner = OfficialTickerScanner()

# Integrate parameters
integrate_trained_parameters(scanner, "trained_parameters/scanner_parameters.json")

# Run scanner
results = scanner.scan(...)
```

## Output Files

### `trained_parameters/scanner_parameters.json`

```json
{
  "metadata": {
    "training_date": "2025-11-04T...",
    "lookback_months": 12,
    "num_tickers": 40,
    "num_trades": 1248
  },
  "priors": {
    "global": {
      "alpha": 677.0,
      "beta": 572.0,
      "p_win": 0.542,
      "avg_win": 0.068,
      "avg_loss": -0.042
    }
  },
  "setup_parameters": {
    "RSI_Oversold_Long": {
      "p_win": 0.58,
      "avg_win": 0.072,
      "avg_loss": -0.038,
      "total_trades": 245,
      "expectancy": 0.026,
      "profit_factor": 2.1
    },
    ...
  }
}
```

### `trained_parameters/trade_results.csv`

Full trade-by-trade results:
```csv
ticker,setup,entry_date,entry_price,exit_date,exit_price,direction,return_pct,r_multiple,is_win,exit_reason
AAPL,RSI_Oversold_Long,2024-05-10,175.23,2024-05-15,181.45,1,0.0355,1.82,True,TARGET
TSLA,MACD_Cross_Long,2024-06-01,178.50,2024-06-08,171.20,-1,-0.0409,-1.15,False,STOP
...
```

## Advanced Features

### 1. Setup-Specific Parameters

The system tracks statistics for each trading setup individually:

```python
from parameter_integration import TrainedParameterLoader

loader = TrainedParameterLoader()
params = loader.get_setup_parameters("RSI_Oversold_Long")

print(f"P(Win): {params['p_win']:.1%}")
print(f"Expectancy: {params['expectancy']:.2%}")
```

### 2. Ticker-Specific Parameters

Train on specific tickers to get ticker-specific priors:

```python
trainer = HistoricalParameterTrainer(
    tickers=['AAPL', 'TSLA', 'NVDA'],
    lookback_months=12
)
trainer.run()

# Access in statistics['by_setup_ticker']
```

### 3. Parameter Monitoring

Track how often trained vs real parameters are used:

```python
from parameter_integration import ParameterMonitor

monitor = ParameterMonitor()

# During scanner operation
monitor.record_usage(setup="RSI_Oversold_Long", used_trained=True, confidence=0.8)

# View summary
monitor.print_summary()
```

### 4. Custom Training Configurations

Customize the training process:

```python
from historical_parameter_trainer import HistoricalParameterTrainer

trainer = HistoricalParameterTrainer(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    lookback_months=24,  # 2 years
    output_dir="custom_parameters"
)

# Customize signal generator
trainer.signal_generator.indicators.calculate_rsi(period=21)  # Use RSI(21)

# Customize trade simulator
trainer.trade_simulator.max_holding_period = 30  # 30-day max hold

trainer.run()
```

### 5. Regime-Specific Training

Train parameters for different volatility regimes:

```python
# The system already tracks regime-specific data
# Access via statistics['by_setup']

# Low volatility: < 15%
# Medium volatility: 15-30%
# High volatility: > 30%
```

## Performance Considerations

### Training Time
- **40 tickers, 12 months**: ~5-10 minutes
- **100 tickers, 24 months**: ~20-30 minutes
- Parallelized data fetching (8 workers)
- Rate limiting prevents API throttling

### Data Requirements
- Minimum 200 bars per ticker (for indicator warmup)
- Recommended: 1+ years of daily data
- More data = more reliable parameters

### Sample Size Requirements
- **Minimum trades per setup**: 30 (for statistical significance)
- **Recommended trades per setup**: 100+
- System automatically falls back to global priors if insufficient data

## Statistical Methodology

### Bayesian Prior Update

The system uses **Beta-Binomial conjugate priors**:

1. **Prior**: Beta(α₀, β₀) where α₀ = β₀ = 1 (uniform prior)
2. **Data**: W wins out of N trades
3. **Posterior**: Beta(α₀ + W, β₀ + N - W)
4. **Estimate**: P(win) = (α₀ + W) / (α₀ + β₀ + N)

This provides:
- Natural uncertainty quantification
- Smooth parameter updates as data accumulates
- Protection against overfitting to small samples

### Return Estimates

**Average Win/Loss**:
```python
avg_win = sum(r for r in returns if r > 0) / count(wins)
avg_loss = sum(r for r in returns if r < 0) / count(losses)
```

**Expectancy**:
```python
expectancy = p_win * avg_win + (1 - p_win) * avg_loss
```

**Profit Factor**:
```python
profit_factor = sum(wins) / abs(sum(losses))
```

## Validation & Backtesting

### Walk-Forward Validation

To validate the approach, use walk-forward testing:

```bash
# Train on first 6 months
python run_parameter_training.py --lookback 6 --output params_train

# Test on next 6 months (TODO: implement test script)
python backtest_parameters.py --params params_train --test-months 6
```

### Cross-Validation

Train on different ticker groups and compare:

```bash
# Train on tech stocks
python run_parameter_training.py --tickers AAPL MSFT GOOGL NVDA --output params_tech

# Train on finance stocks
python run_parameter_training.py --tickers JPM BAC GS MS --output params_finance

# Compare results
python compare_parameters.py params_tech params_finance
```

## Troubleshooting

### Issue: "Insufficient data for ticker"
**Solution**: Increase lookback period or check if ticker has been recently listed

### Issue: "No trained parameters found"
**Solution**: Run `python run_parameter_training.py` first

### Issue: "Too few trades for setup X"
**Solution**: 
- Add more tickers to training set
- Increase lookback period
- Relax signal generation criteria

### Issue: "Trained parameters show negative expectancy"
**Solution**: This is valid! Some setups may not be profitable. The scanner will automatically avoid them.

## Future Enhancements

1. **Real-time Learning**: Update parameters as scanner executes real trades
2. **Regime Detection**: Automatically adjust parameters based on current market regime
3. **Multi-timeframe**: Train on different timeframes (1h, 4h, daily)
4. **Feature Engineering**: Add more sophisticated technical indicators
5. **ML Integration**: Use trained parameters as features for ML models
6. **Risk Parity**: Weight setups by inverse volatility
7. **Portfolio Optimization**: Optimize setup mix for Sharpe ratio

## References

- [Empirical Bayes Methods](https://en.wikipedia.org/wiki/Empirical_Bayes_method)
- [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Beta-Binomial Distribution](https://en.wikipedia.org/wiki/Beta-binomial_distribution)
- [Van Tharp's R-Multiples](https://www.vantharp.com/)

## License

Part of the EVR (Expected Value Ratio) trading framework.

## Support

For issues or questions, check the main EVR documentation or create an issue in the repository.



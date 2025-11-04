# EVR - NYSE & NASDAQ Ticker Scanner

A sophisticated trading signal scanner for NYSE and NASDAQ stocks with historical parameter training, backtesting capabilities, and portfolio management.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train parameters (optional but recommended)
pip install -r requirements_training.txt
python run_parameter_training.py

# Run the scanner
python official_scanner.py
```

## ✨ Features

### Core Scanning
- **Multi-Exchange Support**: Scans NYSE & NASDAQ tickers
- **Technical Analysis**: 6+ trading setups (RSI, MACD, Bollinger Bands, etc.)
- **Signal Aggregation**: Combines multiple signals for higher confidence
- **Real-time Data**: Fetches live market data via yfinance
- **Risk Management**: Calculates position sizing, stop losses, and take profits

### Advanced Features
- **Historical Parameter Training**: Train probabilistic parameters from backtested data
- **Backtesting Engine**: Validate strategies with real historical data
- **Walk-Forward Analysis**: Test strategy robustness over rolling windows
- **Portfolio Management**: Track positions, capital, and returns
- **Position Monitoring**: Real-time tracking of open positions

## 📊 Output Formats

The scanner generates comprehensive reports in multiple formats:
- **CSV**: Spreadsheet-compatible signal data
- **JSON**: API-friendly structured data
- **Summary**: Human-readable text reports

## 📚 Documentation

### Getting Started
- [Get Started Guide](docs/GET_STARTED.md) - Quick 3-command setup
- [Usage Guide](docs/USAGE.md) - Comprehensive usage examples

### Parameter Training
- [Parameter Training README](docs/README_PARAMETERS.md) - Overview of parameter system
- [Quickstart Guide](docs/QUICKSTART_PARAMETER_TRAINING.md) - Fast tutorial
- [Complete Reference](docs/PARAMETER_TRAINING_README.md) - Full documentation

### Implementation Details
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md) - Technical architecture
- [Changes Summary](docs/CHANGES_SUMMARY.md) - Recent updates

### Recent Improvements
- [All Fixes Complete](docs/ALL_FIXES_COMPLETE.md) - Complete list of fixes
- [Fixes Summary Nov 4](docs/FIXES_SUMMARY_NOV4.md) - Latest improvements
- [Auto Position Monitoring](docs/AUTO_POSITION_MONITORING.md) - Position tracking feature
- [Auto Retraining](docs/AUTO_RETRAINING_FEATURE.md) - Automatic parameter updates
- [Capital & Position Size Fix](docs/CAPITAL_AND_POSITION_SIZE_FIX.md) - Portfolio management fix
- [Setup Name Fix](docs/SETUP_NAME_FIX.md) - Signal naming improvements

## 🔧 Installation

### Basic Installation
```bash
# Clone the repository
git clone <repository-url>
cd evr

# Install core dependencies
pip install -r requirements.txt
```

### With Parameter Training
```bash
# Install training dependencies
pip install -r requirements_training.txt
```

## 📖 Usage Examples

### Basic Scanning
```bash
# Default scan (all tickers, top 20 aggregated)
python official_scanner.py

# Limited scan (faster)
python official_scanner.py --max-tickers 100

# Custom output name
python official_scanner.py --output-prefix "my_scan"
```

### Parameter Training
```bash
# Train parameters from historical data
python run_parameter_training.py

# Train on specific tickers
python run_parameter_training.py --tickers AAPL MSFT GOOGL

# Use longer history
python run_parameter_training.py --lookback 24
```

### Backtesting
```bash
# Basic backtest
python official_scanner.py --backtest \
  --start-date 2023-06-01 \
  --end-date 2023-08-01

# Walk-forward analysis
python official_scanner.py --backtest --walk-forward \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --training-window 90 \
  --testing-window 30
```

## 🏗️ Project Structure

```
evr/
├── official_scanner.py           # Main scanner application
├── portfolio_state.json          # Portfolio tracking state
│
├── 🔧 Parameter Training
│   ├── historical_parameter_trainer.py
│   ├── parameter_integration.py
│   ├── run_parameter_training.py
│   ├── demo_parameter_system.py
│   └── test_parameter_system.py
│
├── 📁 Trained Parameters
│   └── trained_parameters/
│       ├── scanner_parameters.json
│       ├── trained_statistics.json
│       └── trade_results.csv
│
├── 📁 Cache
│   └── cache/
│       ├── official_tickers.json
│       └── delisted_tickers.json
│
├── 📁 Scan Results
│   └── scans/
│       ├── *.csv
│       ├── *.json
│       └── *_summary.txt
│
├── 📚 Documentation
│   └── docs/
│       ├── GET_STARTED.md
│       ├── USAGE.md
│       ├── README_PARAMETERS.md
│       └── ... (more documentation)
│
└── 📦 Requirements
    ├── requirements.txt
    └── requirements_training.txt
```

## 🎯 Trading Setups

The scanner identifies signals from multiple technical setups:

1. **RSI Oversold Long** - Buy dips when RSI < 30
2. **RSI Overbought Short** - Sell rallies when RSI > 70
3. **MACD Crossover Long** - Momentum entry on bullish cross
4. **Bollinger Band Bounce** - Mean reversion plays
5. **Trend Following Long** - Ride strong trends
6. **Mean Reversion Short** - Fade extremes

Each setup calculates:
- Entry price
- Stop loss level
- Take profit target
- Position size
- Expected return
- Risk/reward ratio

## 📈 Performance Metrics

The scanner and backtesting engine provide comprehensive metrics:

### Trading Metrics
- Win rate & profit factor
- Average win/loss
- Expectancy per trade
- Total trades & returns

### Portfolio Metrics
- Total return & annualized return
- Maximum drawdown
- Sharpe ratio
- Sortino ratio
- Calmar ratio

### Risk Metrics
- Value at Risk (VaR)
- Volatility & downside deviation
- Beta & correlation
- Position sizing

## 🔄 Adaptive Learning

The scanner learns from historical data and adapts to real trades:

**Initial** → Uses trained parameters from backtests  
**15 trades** → Blends 50% trained + 50% real data  
**30+ trades** → Uses 100% real trading data

This ensures:
✓ Good starting point  
✓ Smooth transition  
✓ Eventual adaptation to reality  

## 🛠️ Utilities

### Testing
```bash
# Test parameter system
python test_parameter_system.py

# Demo parameter system
python demo_parameter_system.py
```

### Analysis
```bash
# Analyze existing training results
python run_parameter_training.py --mode analyze

# View backtest results
cat scans/*_backtest_*_summary.txt
```

## 📝 Best Practices

1. **Regular Training**: Retrain parameters monthly as markets evolve
2. **Diverse Universe**: Use 20-50 diverse tickers for training
3. **Historical Depth**: Include 12+ months of history
4. **Validation**: Use walk-forward analysis for robust testing
5. **Monitoring**: Track real vs predicted performance
6. **Risk Management**: Always use stop losses and position sizing

## 🔍 Troubleshooting

### Common Issues

**No signals generated:**
```bash
python official_scanner.py --max-tickers 10 --log-level DEBUG
```

**Slow performance:**
```bash
python official_scanner.py --max-tickers 50 --log-level WARNING
```

**Module not found:**
```bash
pip install -r requirements.txt
pip install -r requirements_training.txt
```

**No trained parameters:**
```bash
python run_parameter_training.py
```

## 📊 Data Sources

- **Market Data**: Yahoo Finance (yfinance)
- **Ticker Lists**: NASDAQ FTP & official exchange lists
- **Delisted Tracking**: Automated cache management

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading involves risk of loss. Always do your own research and consider consulting with a licensed financial advisor before making investment decisions.

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Tests pass
- Documentation is updated
- Commits are descriptive

## 📞 Support

For issues, questions, or suggestions:
1. Check the [documentation](docs/)
2. Review [troubleshooting guides](docs/USAGE.md#troubleshooting)
3. Open an issue on the repository

---

**Ready to get started?** → [Get Started Guide](docs/GET_STARTED.md)


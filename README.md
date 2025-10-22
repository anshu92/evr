# EVR - Empirical Volatility Regime Trading Framework

EVR is a quantitative trading framework designed for empirical volatility regime analysis and systematic trading strategy development.

## Features

- **Data Management**: Multi-source data adapters with caching (Yahoo Finance, Stooq)
- **Feature Engineering**: Comprehensive technical indicator library with lazy computation
- **Trading Setups**: Built-in setups including Squeeze Breakout, Trend Pullback, and Mean Reversion
- **Probability Models**: Rolling Bayesian probability estimation with regime detection
- **Risk Management**: Kelly criterion position sizing with circuit breakers
- **Backtesting**: Vectorized backtesting engine with realistic trade execution
- **Recommendations**: Real-time scanning and recommendation engine
- **Reporting**: Comprehensive HTML/Markdown reports with charts

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Basic Configuration

Create a configuration file (`config.yaml`):

```yaml
name: "My EVR Strategy"
description: "Custom trading strategy"

data:
  primary_source: "yfinance"
  cache_dir: "~/.evr/data"
  timeframe: "1d"

risk:
  initial_capital: 100000.0
  max_position_size: 0.1
  max_risk_per_trade: 0.02

prob:
  window_size: 252
  min_samples: 50
```

### 2. Command Line Usage

#### Scan for Recommendations

```bash
evr scan --config config.yaml --symbols AAPL,MSFT,GOOGL --top 10
```

#### Run Backtest

```bash
evr backtest --config config.yaml --start 2020-01-01 --end 2025-01-01 --symbols AAPL,MSFT,GOOGL
```

#### Generate Report

```bash
evr report <run-id> --config config.yaml
```

### 3. Python API Usage

#### Minimal Scan Example

```python
from evr.config import load_config
from evr.recommend import RecommendationScanner

# Load configuration
config = load_config("config.yaml")

# Initialize scanner
scanner = RecommendationScanner(config)

# Run scan
trade_plans = scanner.scan(
    symbols=["AAPL", "MSFT", "GOOGL"],
    setups=["squeeze_breakout", "trend_pullback", "mean_reversion"],
    top_n=5,
)

# Display results
for plan in trade_plans:
    print(f"{plan.signal.symbol}: {plan.signal.setup} - {plan.probability:.1%} win probability")
```

#### Minimal Backtest Example

```python
from evr.config import load_config
from evr.backtest import BacktestEngine

# Load configuration
config = load_config("config.yaml")

# Initialize backtest engine
engine = BacktestEngine(config)

# Run backtest
results = engine.run_backtest(
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2025-01-01",
    setups=["squeeze_breakout", "trend_pullback", "mean_reversion"],
)

# Display results
metrics = results['metrics']
print(f"Total Return: {metrics.total_return:.2%}")
print(f"CAGR: {metrics.cagr:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
```

## Built-in Trading Setups

### 1. Squeeze Breakout

Identifies periods of low volatility (squeeze) followed by breakouts using Bollinger Bands and Keltner Channels.

**Key Features:**
- Detects squeeze conditions when BB is inside KC
- Identifies breakouts with volume confirmation
- Configurable squeeze duration and breakout thresholds

### 2. Trend Pullback

Identifies pullbacks in trending markets and looks for continuation signals using moving averages and trend strength indicators.

**Key Features:**
- Uses fast/slow moving averages to identify trends
- Detects pullbacks and reversal signals
- Configurable trend strength and pullback parameters

### 3. Mean Reversion

Identifies overbought/oversold conditions using RSI, Bollinger Bands, and Z-score indicators.

**Key Features:**
- Multi-indicator confirmation (RSI, BB, Z-score)
- Configurable overbought/oversold levels
- Volume and price action confirmation

## Configuration Reference

### Data Configuration

```yaml
data:
  primary_source: "yfinance"          # Primary data source
  fallback_sources: ["stooq"]         # Fallback sources
  cache_dir: "~/.evr/data"           # Cache directory
  cache_format: "parquet"            # Cache format
  cache_ttl_days: 1                  # Cache TTL in days
  timeframe: "1d"                    # Data timeframe
  adjust_splits: true                # Adjust for stock splits
  adjust_dividends: true             # Adjust for dividends
  timezone: "America/Toronto"        # Timezone
```

### Risk Configuration

```yaml
risk:
  initial_capital: 100000.0          # Initial capital
  max_position_size: 0.1             # Max position size (10%)
  max_sector_exposure: 0.3           # Max sector exposure (30%)
  max_correlation: 0.7               # Max correlation between positions
  max_risk_per_trade: 0.02           # Max risk per trade (2%)
  min_risk_per_trade: 0.005          # Min risk per trade (0.5%)
  use_kelly: true                    # Use Kelly criterion
  kelly_fraction: 0.25               # Kelly fraction multiplier
  max_kelly_fraction: 0.5            # Max Kelly fraction cap
  daily_loss_limit: 0.05             # Daily loss limit (5%)
  weekly_loss_limit: 0.15             # Weekly loss limit (15%)
  max_drawdown_limit: 0.25           # Max drawdown limit (25%)
  max_positions: 20                  # Max number of positions
  min_positions: 1                   # Min number of positions
```

### Probability Configuration

```yaml
prob:
  window_size: 252                   # Rolling window size (trading days)
  min_samples: 50                    # Minimum samples for estimation
  alpha: 0.1                         # EWMA smoothing factor
  beta_prior_alpha: 1.0              # Beta distribution alpha
  beta_prior_beta: 1.0               # Beta distribution beta
  use_regimes: false                 # Use regime detection
  regime_window: 63                  # Regime detection window
  regime_threshold: 0.1              # Regime change threshold
  update_frequency: "daily"          # Update frequency
```

## Architecture

### Core Components

1. **Data Layer**: Multi-source data adapters with caching
2. **Feature Pipeline**: Technical indicator computation with lazy evaluation
3. **Setup Engine**: Pluggable trading setup system
4. **Probability Models**: Bayesian probability estimation
5. **Risk Management**: Kelly sizing and circuit breakers
6. **Backtesting Engine**: Vectorized backtesting with realistic execution
7. **Recommendation Engine**: Real-time scanning and ranking
8. **Reporting System**: Comprehensive reports with charts

### Data Flow

```
Data Sources → Data Adapters → Cache → Feature Pipeline → Setups → Signals → Trade Plans → Risk Management → Execution → Results → Reporting
```

## Development

### Project Structure

```
evr/
├── evr/                    # Main package
│   ├── __init__.py
│   ├── types.py            # Core types and dataclasses
│   ├── config/             # Configuration management
│   ├── data/               # Data layer
│   ├── features/           # Feature engineering
│   ├── setups/             # Trading setups
│   ├── models/             # Probability and payoff models
│   ├── risk/               # Risk management
│   ├── backtest/           # Backtesting engine
│   ├── recommend/          # Recommendation engine
│   └── reporting/          # Reporting system
├── cli.py                  # CLI interface
├── examples/               # Example scripts
├── tests/                  # Test suite
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Linting
ruff check evr/

# Type checking
mypy evr/

# Formatting
ruff format evr/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It is not intended for use in live trading without proper testing and validation. Always verify results and understand the risks before using any trading strategy.

## Support

For questions, issues, or contributions, please use the GitHub repository's issue tracker.

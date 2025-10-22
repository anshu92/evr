# EVR Scanner Usage Guide

This guide provides practical examples and use cases for the EVR NYSE & NASDAQ Ticker Scanner.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Backtesting](#backtesting)
5. [Output Formats](#output-formats)
6. [Performance Tips](#performance-tips)
7. [Common Use Cases](#common-use-cases)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd evr

# Install dependencies
uv venv
source .venv/bin/activate
uv pip install pandas numpy pyarrow yfinance requests rich lxml html5lib
uv pip install -e .
```

### 2. First Run
```bash
# Basic scan with aggregated recommendations (scans all tickers by default)
python official_scanner.py

# Individual signals mode
python official_scanner.py --no-aggregate

# Quick backtest
python official_scanner.py --backtest --start-date 2023-06-01 --end-date 2023-08-01 --max-tickers 10
```

## Basic Usage

### Default Scan
```bash
# Scan all tickers, show top 20 aggregated recommendations
python official_scanner.py
```

### Custom Parameters
```bash
# Limit to specific number of tickers
python official_scanner.py --max-tickers 200

# Show more results
python official_scanner.py --top 50

# Custom output name
python official_scanner.py --output-prefix "my_scan"
```

### Logging Levels
```bash
# Quiet mode (errors only)
python official_scanner.py --log-level ERROR

# Normal mode (default)
python official_scanner.py --log-level INFO

# Verbose mode (debug info)
python official_scanner.py --log-level DEBUG
```

## Advanced Usage

### Production Scans
```bash
# Large comprehensive scan
python official_scanner.py --max-tickers 500 --log-level INFO --output-prefix "production_scan"

# Daily automated scan
python official_scanner.py --max-tickers 300 --output-prefix "daily_$(date +%Y%m%d)"
```

### Testing and Development
```bash
# Quick test scan
python official_scanner.py --max-tickers 25 --top 10 --log-level DEBUG --output-prefix "test"

# Fresh data scan
python official_scanner.py --no-cache --max-tickers 50 --output-prefix "fresh_data"
```

### Batch Processing
```bash
# Process in batches
for i in {1..5}; do
    python official_scanner.py --max-tickers 200 --output-prefix "batch_$i"
    sleep 60  # Wait 1 minute between batches
done
```

## Backtesting

The EVR scanner includes a comprehensive backtesting engine that validates strategy performance using real historical data, transaction costs, and statistical metrics.

### Basic Backtesting

#### Single Backtest
```bash
# Basic backtest with default parameters
python official_scanner.py --backtest --start-date 2023-06-01 --end-date 2023-08-01

# Custom capital and positions
python official_scanner.py --backtest \
  --start-date 2023-06-01 \
  --end-date 2023-08-01 \
  --backtest-capital 100000 \
  --max-positions 5

# Different rebalancing frequency
python official_scanner.py --backtest \
  --start-date 2023-06-01 \
  --end-date 2023-08-01 \
  --rebalance-frequency daily
```

#### Walk-Forward Analysis
```bash
# Walk-forward analysis with rolling windows
python official_scanner.py --backtest --walk-forward \
  --start-date 2023-06-01 \
  --end-date 2023-08-01 \
  --training-window 30 \
  --testing-window 15 \
  --backtest-capital 100000

# Longer training window for more stable results
python official_scanner.py --backtest --walk-forward \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --training-window 90 \
  --testing-window 30
```

### Backtest Parameters

#### Core Parameters
- `--backtest`: Enable backtesting mode
- `--start-date`: Start date for backtest (YYYY-MM-DD)
- `--end-date`: End date for backtest (YYYY-MM-DD)
- `--backtest-capital`: Initial capital (default: $100,000)
- `--max-positions`: Maximum concurrent positions (default: 10)
- `--rebalance-frequency`: Rebalancing frequency (daily/weekly/monthly)

#### Walk-Forward Parameters
- `--walk-forward`: Enable walk-forward analysis
- `--training-window`: Training window in days (default: 90)
- `--testing-window`: Testing window in days (default: 30)

### Backtest Results

#### Performance Metrics
The backtest generates comprehensive performance metrics:

**Core Metrics:**
- **Total Return**: Overall portfolio return
- **Annualized Return**: Yearly return rate
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of trades executed

**Transaction Costs:**
- **Total Costs**: Commission + slippage costs
- **Cost Ratio**: Costs as percentage of initial capital

**Benchmark Comparison:**
- **Benchmark Return**: S&P 500 (SPY) performance
- **Outperformance**: Strategy vs benchmark excess return

#### Statistical Validation Metrics
- **Calmar Ratio**: Annualized return / maximum drawdown
- **Information Ratio**: Excess return / tracking error
- **Sortino Ratio**: Return / downside deviation
- **Value at Risk (VaR)**: 95% and 99% VaR calculations
- **Volatility**: Daily return volatility
- **Downside Deviation**: Downside volatility

### Backtest Examples

#### 1. Strategy Validation
```bash
# Validate strategy over 6 months
python official_scanner.py --backtest \
  --start-date 2023-01-01 \
  --end-date 2023-06-30 \
  --backtest-capital 50000 \
  --max-positions 3 \
  --output-prefix "strategy_validation"

# Review results
python -c "
import json
with open('scans/strategy_validation_backtest_*.json') as f:
    results = json.load(f)
    metrics = results['metrics']
    print(f'Total Return: {metrics[\"total_return\"]:.2%}')
    print(f'Sharpe Ratio: {metrics[\"sharpe_ratio\"]:.3f}')
    print(f'Max Drawdown: {metrics[\"max_drawdown\"]:.2%}')
    print(f'Win Rate: {metrics[\"win_rate\"]:.2%}')
"
```

#### 2. Parameter Optimization
```bash
# Test different position sizes
for positions in 3 5 10 15; do
    python official_scanner.py --backtest \
      --start-date 2023-06-01 \
      --end-date 2023-08-01 \
      --max-positions $positions \
      --output-prefix "positions_${positions}"
done

# Compare results
python -c "
import glob
import json
import pandas as pd

results = []
for file in glob.glob('scans/positions_*_backtest_*.json'):
    with open(file) as f:
        data = json.load(f)
        metrics = data['metrics']
        positions = file.split('positions_')[1].split('_')[0]
        results.append({
            'positions': int(positions),
            'return': metrics['total_return'],
            'sharpe': metrics['sharpe_ratio'],
            'drawdown': metrics['max_drawdown']
        })

df = pd.DataFrame(results)
print('Position Size Optimization:')
print(df.sort_values('sharpe', ascending=False))
"
```

#### 3. Market Regime Analysis
```bash
# Test during different market conditions
python official_scanner.py --backtest \
  --start-date 2023-01-01 \
  --end-date 2023-03-31 \
  --output-prefix "q1_2023"

python official_scanner.py --backtest \
  --start-date 2023-04-01 \
  --end-date 2023-06-30 \
  --output-prefix "q2_2023"

python official_scanner.py --backtest \
  --start-date 2023-07-01 \
  --end-date 2023-09-30 \
  --output-prefix "q3_2023"

# Analyze performance across quarters
python -c "
import glob
import json
import pandas as pd

quarters = []
for file in glob.glob('scans/q*_2023_backtest_*.json'):
    with open(file) as f:
        data = json.load(f)
        metrics = data['metrics']
        quarter = file.split('_')[0].split('/')[-1]
        quarters.append({
            'quarter': quarter,
            'return': metrics['total_return'],
            'sharpe': metrics['sharpe_ratio'],
            'drawdown': metrics['max_drawdown'],
            'trades': metrics['total_trades']
        })

df = pd.DataFrame(quarters)
print('Quarterly Performance:')
print(df)
"
```

#### 4. Walk-Forward Validation
```bash
# Comprehensive walk-forward analysis
python official_scanner.py --backtest --walk-forward \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --training-window 60 \
  --testing-window 20 \
  --backtest-capital 100000 \
  --output-prefix "walkforward_2023"

# Analyze walk-forward results
python -c "
import json
with open('scans/walkforward_2023_walkforward_*.json') as f:
    results = json.load(f)
    aggregate = results['aggregate_metrics']
    
    print('Walk-Forward Analysis Results:')
    print(f'Total Periods: {aggregate[\"total_periods\"]}')
    print(f'Average Return: {aggregate[\"avg_return\"]:.2%}')
    print(f'Return Volatility: {aggregate[\"return_std\"]:.2%}')
    print(f'Average Sharpe: {aggregate[\"avg_sharpe\"]:.3f}')
    print(f'Consistency: {aggregate[\"consistency\"]:.2%}')
    print(f'Positive Periods: {aggregate[\"positive_periods\"]}')
"
```

### Backtest Output Files

#### JSON Results
```bash
# Detailed backtest results in JSON format
python official_scanner.py --backtest --output-prefix "detailed_backtest"

# Parse JSON results
python -c "
import json
with open('scans/detailed_backtest_backtest_*.json') as f:
    results = json.load(f)
    
    # Portfolio details
    portfolio = results['portfolio']
    print(f'Final Value: ${portfolio[\"total_value\"]:,.0f}')
    print(f'Cash: ${portfolio[\"cash\"]:,.0f}')
    print(f'Positions: {len(portfolio[\"positions\"])}')
    
    # Trade details
    trades = portfolio['trades']
    print(f'\\nTrade Summary:')
    for trade in trades[-5:]:  # Last 5 trades
        print(f'{trade[\"date\"]}: {trade[\"ticker\"]} {trade[\"action\"]} - P&L: ${trade.get(\"net_pnl\", 0):.2f}')
"
```

#### Summary Reports
```bash
# Generate summary report
python official_scanner.py --backtest --output-prefix "summary_backtest"

# View summary
cat scans/summary_backtest_backtest_*_summary.txt
```

### Backtest Best Practices

#### 1. Realistic Parameters
```bash
# Use realistic capital and position sizes
python official_scanner.py --backtest \
  --backtest-capital 100000 \
  --max-positions 5 \
  --rebalance-frequency weekly
```

#### 2. Multiple Time Periods
```bash
# Test across different market conditions
python official_scanner.py --backtest \
  --start-date 2022-01-01 \
  --end-date 2022-12-31 \
  --output-prefix "backtest_2022"

python official_scanner.py --backtest \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --output-prefix "backtest_2023"
```

#### 3. Walk-Forward Validation
```bash
# Always validate with walk-forward analysis
python official_scanner.py --backtest --walk-forward \
  --start-date 2022-01-01 \
  --end-date 2023-12-31 \
  --training-window 90 \
  --testing-window 30
```

#### 4. Benchmark Comparison
```bash
# Compare against market benchmark
python official_scanner.py --backtest \
  --start-date 2023-06-01 \
  --end-date 2023-08-01

# Results will show SPY benchmark comparison automatically
```

### Backtest Troubleshooting

#### Common Issues

**1. No Trades Generated**
```bash
# Debug with verbose logging
python official_scanner.py --backtest \
  --start-date 2023-06-01 \
  --end-date 2023-08-01 \
  --max-tickers 10 \
  --log-level DEBUG
```

**2. Historical Data Issues**
```bash
# Test with known good tickers
python official_scanner.py --backtest \
  --start-date 2023-06-01 \
  --end-date 2023-08-01 \
  --max-tickers 5 \
  --backtest-capital 10000
```

**3. Performance Issues**
```bash
# Use smaller date ranges for testing
python official_scanner.py --backtest \
  --start-date 2023-07-01 \
  --end-date 2023-07-31 \
  --max-tickers 10
```

#### Debug Commands
```bash
# Full debug backtest
python official_scanner.py --backtest \
  --start-date 2023-06-01 \
  --end-date 2023-08-01 \
  --max-tickers 3 \
  --log-level DEBUG \
  --output-prefix "debug_backtest"

# Check historical data availability
python -c "
import yfinance as yf
ticker = yf.Ticker('AAPL')
data = ticker.history(start='2023-06-01', end='2023-08-01')
print(f'AAPL data points: {len(data)}')
print(f'Date range: {data.index[0]} to {data.index[-1]}')
"
```

## Output Formats

### CSV Output
```bash
# Generate CSV for spreadsheet analysis
python official_scanner.py --output-prefix "spreadsheet_data"

# View top 10 signals
head -11 scans/spreadsheet_data_*.csv
```

### JSON Output
```bash
# Generate JSON for API integration
python official_scanner.py --output-prefix "api_data"

# Parse JSON output
python -c "
import json
with open('scans/api_data_*.json') as f:
    data = json.load(f)
    print(f'Found {len(data)} signals')
    for signal in data[:3]:
        print(f'{signal[\"ticker\"]}: {signal[\"signal_type\"]} - {signal[\"direction\"]}')
"
```

### Summary Output
```bash
# Generate summary for quick review
python official_scanner.py --output-prefix "summary"

# View summary
cat scans/summary_*_summary.txt
```

## Performance Tips

### Optimize Scanning Speed
```bash
# Faster scanning (fewer tickers)
python official_scanner.py --max-tickers 50 --log-level WARNING

# Balanced approach
python official_scanner.py --max-tickers 100 --log-level INFO

# Comprehensive but slower
python official_scanner.py --max-tickers 500 --log-level INFO
```

### Memory Management
```bash
# Small batches for limited memory
python official_scanner.py --max-tickers 25

# Medium batches
python official_scanner.py --max-tickers 100

# Large batches (requires more memory)
python official_scanner.py --max-tickers 500
```

### Network Optimization
```bash
# Use cached data when possible
python official_scanner.py  # Uses cache by default

# Force fresh data only when needed
python official_scanner.py --no-cache
```

## Common Use Cases

### 1. Daily Trading Signals
```bash
#!/bin/bash
# daily_signals.sh
cd /path/to/evr
source .venv/bin/activate

# Generate daily signals
python official_scanner.py \
    --max-tickers 200 \
    --top 30 \
    --output-prefix "daily_$(date +%Y%m%d)" \
    --log-level INFO

# Email results (if configured)
# mail -s "Daily Trading Signals" trader@example.com < scans/daily_*_summary.txt
```

### 2. Portfolio Screening
```bash
# Screen for long opportunities
python official_scanner.py --max-tickers 300 --output-prefix "long_screen"

# Filter for high-confidence signals
python -c "
import pandas as pd
df = pd.read_csv('scans/long_screen_*.csv')
high_conf = df[df['confidence'] > 0.7]
print(f'High confidence signals: {len(high_conf)}')
print(high_conf[['ticker', 'signal_type', 'confidence', 'expected_return']].head())
"
```

### 3. Risk Assessment
```bash
# Generate signals for risk analysis
python official_scanner.py --max-tickers 150 --output-prefix "risk_analysis"

# Analyze risk-reward ratios
python -c "
import pandas as pd
df = pd.read_csv('scans/risk_analysis_*.csv')
print('Risk-Reward Analysis:')
print(f'Average R/R: {df[\"risk_reward_ratio\"].mean():.2f}')
print(f'Min R/R: {df[\"risk_reward_ratio\"].min():.2f}')
print(f'Max R/R: {df[\"risk_reward_ratio\"].max():.2f}')
print(f'Signals with R/R > 2.0: {len(df[df[\"risk_reward_ratio\"] > 2.0])}')
"
```

### 4. Signal Validation
```bash
# Generate signals for validation
python official_scanner.py --max-tickers 100 --log-level DEBUG --output-prefix "validation"

# Check signal distribution
python -c "
import pandas as pd
df = pd.read_csv('scans/validation_*.csv')
print('Signal Distribution:')
print(df['signal_type'].value_counts())
print(f'\\nDirection Distribution:')
print(df['direction'].value_counts())
print(f'\\nConfidence Stats:')
print(f'Mean: {df[\"confidence\"].mean():.3f}')
print(f'Std: {df[\"confidence\"].std():.3f}')
"
```

### 5. Market Analysis
```bash
# Generate comprehensive market signals
python official_scanner.py --max-tickers 500 --output-prefix "market_analysis"

# Analyze market sentiment
python -c "
import pandas as pd
df = pd.read_csv('scans/market_analysis_*.csv')
long_signals = len(df[df['direction'] == 'LONG'])
short_signals = len(df[df['direction'] == 'SHORT'])
total_signals = len(df)

print(f'Market Sentiment Analysis:')
print(f'Total Signals: {total_signals}')
print(f'Long Signals: {long_signals} ({long_signals/total_signals:.1%})')
print(f'Short Signals: {short_signals} ({short_signals/total_signals:.1%})')
print(f'Bullish Bias: {long_signals/total_signals:.1%}')
"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. No Signals Generated
```bash
# Debug mode to see what's happening
python official_scanner.py --max-tickers 10 --log-level DEBUG

# Check if tickers have data
python -c "
import yfinance as yf
ticker = yf.Ticker('AAPL')
data = ticker.history(period='1y')
print(f'AAPL data points: {len(data)}')
print(f'Last price: {data[\"Close\"].iloc[-1]:.2f}')
"
```

#### 2. Slow Performance
```bash
# Reduce ticker count
python official_scanner.py --max-tickers 50

# Use warning level logging
python official_scanner.py --log-level WARNING

# Check system resources
top -l 1 | grep python
```

#### 3. Memory Issues
```bash
# Use smaller batches
python official_scanner.py --max-tickers 25

# Clear cache if needed
rm cache/official_tickers.json

# Monitor memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent:.1f}%')
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"
```

#### 4. Network Issues
```bash
# Test network connectivity
ping ftp.nasdaqtrader.com

# Use cached data
python official_scanner.py  # Uses cache by default

# Check if cache exists
ls -la cache/official_tickers.json
```

### Debug Commands
```bash
# Full debug scan
python official_scanner.py --max-tickers 5 --log-level DEBUG --output-prefix "debug"

# Test specific ticker
python -c "
import yfinance as yf
ticker = yf.Ticker('AAPL')
data = ticker.history(period='1y')
print(f'Data shape: {data.shape}')
print(f'Columns: {list(data.columns)}')
print(f'Date range: {data.index[0]} to {data.index[-1]}')
"
```

### Performance Monitoring
```bash
# Time the scan
time python official_scanner.py --max-tickers 100

# Monitor system resources
python -c "
import psutil
import time
start_time = time.time()
# Run scan here
end_time = time.time()
print(f'Scan time: {end_time - start_time:.2f} seconds')
print(f'CPU usage: {psutil.cpu_percent():.1f}%')
print(f'Memory usage: {psutil.virtual_memory().percent:.1f}%')
"
```

## Integration Examples

### With EVR Backtesting
```bash
# Generate signals for backtesting
python official_scanner.py --max-tickers 50 --output-prefix "backtest_signals"

# Extract tickers for backtesting
python -c "
import pandas as pd
df = pd.read_csv('scans/backtest_signals_*.csv')
tickers = df['ticker'].tolist()
print(' '.join(tickers))
" > tickers.txt

# Run backtest with extracted tickers
python official_scanner.py --backtest \
  --start-date 2023-06-01 \
  --end-date 2023-08-01 \
  --max-tickers 50 \
  --output-prefix "extracted_backtest"

# Compare signal generation vs backtest results
python -c "
import pandas as pd
import json
import glob

# Load signals
signals_df = pd.read_csv('scans/backtest_signals_*.csv')
print(f'Generated {len(signals_df)} signals')

# Load backtest results
backtest_file = glob.glob('scans/extracted_backtest_backtest_*.json')[0]
with open(backtest_file) as f:
    backtest_results = json.load(f)
    
print(f'Backtest trades: {backtest_results[\"metrics\"][\"total_trades\"]}')
print(f'Backtest return: {backtest_results[\"metrics\"][\"total_return\"]:.2%}')
"
```

### With Trading Platforms
```bash
# Generate signals for trading platform
python official_scanner.py --max-tickers 100 --output-prefix "trading_signals"

# Convert to trading platform format
python -c "
import pandas as pd
df = pd.read_csv('scans/trading_signals_*.csv')
# Filter for high confidence signals
high_conf = df[df['confidence'] > 0.6]
# Export in trading platform format
high_conf[['ticker', 'direction', 'entry_price', 'stop_loss', 'take_profit']].to_csv('trading_platform.csv', index=False)
print(f'Exported {len(high_conf)} high-confidence signals')
"
```

### With Alert Systems
```bash
# Generate signals for alerts
python official_scanner.py --max-tickers 200 --output-prefix "alerts"

# Create alert list
python -c "
import pandas as pd
df = pd.read_csv('scans/alerts_*.csv')
# Filter for high expected return
high_return = df[df['expected_return'] > 0.08]
print('High Expected Return Signals:')
for _, signal in high_return.iterrows():
    print(f'{signal[\"ticker\"]}: {signal[\"signal_type\"]} - Expected Return: {signal[\"expected_return\"]:.2%}')
"
```

## Best Practices

### 1. Regular Scans
```bash
# Set up cron job for daily scans
# 0 9 * * 1-5 cd /path/to/evr && source .venv/bin/activate && python official_scanner.py --max-tickers 200 --output-prefix "daily_$(date +%Y%m%d)"
```

### 2. Data Management
```bash
# Clean old scan results weekly
find scans/ -name "*.csv" -mtime +7 -delete
find scans/ -name "*.json" -mtime +7 -delete
find scans/ -name "*.txt" -mtime +7 -delete
```

### 3. Monitoring
```bash
# Check scan results regularly
python -c "
import glob
import pandas as pd
files = glob.glob('scans/*.csv')
if files:
    latest = max(files, key=lambda x: os.path.getmtime(x))
    df = pd.read_csv(latest)
    print(f'Latest scan: {latest}')
    print(f'Signals: {len(df)}')
    print(f'Average confidence: {df[\"confidence\"].mean():.1%}')
else:
    print('No scan results found')
"
```

### 4. Backup
```bash
# Backup important scan results
tar -czf scan_backup_$(date +%Y%m%d).tar.gz scans/
```

### 5. Backtesting Workflow
```bash
# Regular backtesting schedule
# Weekly backtest validation
python official_scanner.py --backtest \
  --start-date $(date -d '1 week ago' +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d) \
  --output-prefix "weekly_validation_$(date +%Y%m%d)"

# Monthly comprehensive backtest
python official_scanner.py --backtest --walk-forward \
  --start-date $(date -d '3 months ago' +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d) \
  --training-window 60 \
  --testing-window 20 \
  --output-prefix "monthly_walkforward_$(date +%Y%m%d)"
```

### 6. Performance Monitoring
```bash
# Monitor backtest performance over time
python -c "
import glob
import json
import pandas as pd
from datetime import datetime

# Collect all backtest results
results = []
for file in glob.glob('scans/*_backtest_*.json'):
    try:
        with open(file) as f:
            data = json.load(f)
            metrics = data['metrics']
            # Extract date from filename
            date_str = file.split('_')[-1].split('.')[0][:8]
            date = datetime.strptime(date_str, '%Y%m%d')
            
            results.append({
                'date': date,
                'return': metrics['total_return'],
                'sharpe': metrics['sharpe_ratio'],
                'drawdown': metrics['max_drawdown'],
                'trades': metrics['total_trades']
            })
    except:
        continue

if results:
    df = pd.DataFrame(results)
    df = df.sort_values('date')
    
    print('Backtest Performance Over Time:')
    print(f'Average Return: {df[\"return\"].mean():.2%}')
    print(f'Average Sharpe: {df[\"sharpe\"].mean():.3f}')
    print(f'Average Drawdown: {df[\"drawdown\"].mean():.2%}')
    print(f'Total Backtests: {len(df)}')
    
    # Show recent performance
    print('\\nRecent Performance (Last 5):')
    print(df.tail()[['date', 'return', 'sharpe', 'drawdown']].to_string(index=False))
else:
    print('No backtest results found')
"
```

This usage guide provides practical examples and workflows for using the EVR scanner effectively in various scenarios, including comprehensive backtesting capabilities.

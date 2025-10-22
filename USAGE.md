# EVR Scanner Usage Guide

This guide provides practical examples and use cases for the EVR NYSE & NASDAQ Ticker Scanner.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Output Formats](#output-formats)
5. [Performance Tips](#performance-tips)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd evr

# Install dependencies
uv venv
source .venv/bin/activate
uv pip install pandas numpy pyarrow yfinance requests rich matplotlib jinja2 seaborn lxml html5lib
uv pip install -e .
```

### 2. First Run
```bash
# Basic scan
python official_scanner.py

# CLI interface
python cli_official_scanner.py
```

## Basic Usage

### Default Scan
```bash
# Scan 100 tickers, show top 20 signals
python cli_official_scanner.py
```

### Custom Parameters
```bash
# Scan more tickers
python cli_official_scanner.py --max-tickers 200

# Show more results
python cli_official_scanner.py --top 50

# Custom output name
python cli_official_scanner.py --output-prefix "my_scan"
```

### Logging Levels
```bash
# Quiet mode (errors only)
python cli_official_scanner.py --log-level ERROR

# Normal mode (default)
python cli_official_scanner.py --log-level INFO

# Verbose mode (debug info)
python cli_official_scanner.py --log-level DEBUG
```

## Advanced Usage

### Production Scans
```bash
# Large comprehensive scan
python cli_official_scanner.py --max-tickers 500 --log-level INFO --output-prefix "production_scan"

# Daily automated scan
python cli_official_scanner.py --max-tickers 300 --output-prefix "daily_$(date +%Y%m%d)"
```

### Testing and Development
```bash
# Quick test scan
python cli_official_scanner.py --max-tickers 25 --top 10 --log-level DEBUG --output-prefix "test"

# Fresh data scan
python cli_official_scanner.py --no-cache --max-tickers 50 --output-prefix "fresh_data"
```

### Batch Processing
```bash
# Process in batches
for i in {1..5}; do
    python cli_official_scanner.py --max-tickers 200 --output-prefix "batch_$i"
    sleep 60  # Wait 1 minute between batches
done
```

## Output Formats

### CSV Output
```bash
# Generate CSV for spreadsheet analysis
python cli_official_scanner.py --output-prefix "spreadsheet_data"

# View top 10 signals
head -11 scans/spreadsheet_data_*.csv
```

### JSON Output
```bash
# Generate JSON for API integration
python cli_official_scanner.py --output-prefix "api_data"

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
python cli_official_scanner.py --output-prefix "summary"

# View summary
cat scans/summary_*_summary.txt
```

## Performance Tips

### Optimize Scanning Speed
```bash
# Faster scanning (fewer tickers)
python cli_official_scanner.py --max-tickers 50 --log-level WARNING

# Balanced approach
python cli_official_scanner.py --max-tickers 100 --log-level INFO

# Comprehensive but slower
python cli_official_scanner.py --max-tickers 500 --log-level INFO
```

### Memory Management
```bash
# Small batches for limited memory
python cli_official_scanner.py --max-tickers 25

# Medium batches
python cli_official_scanner.py --max-tickers 100

# Large batches (requires more memory)
python cli_official_scanner.py --max-tickers 500
```

### Network Optimization
```bash
# Use cached data when possible
python cli_official_scanner.py  # Uses cache by default

# Force fresh data only when needed
python cli_official_scanner.py --no-cache
```

## Common Use Cases

### 1. Daily Trading Signals
```bash
#!/bin/bash
# daily_signals.sh
cd /path/to/evr
source .venv/bin/activate

# Generate daily signals
python cli_official_scanner.py \
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
python cli_official_scanner.py --max-tickers 300 --output-prefix "long_screen"

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
python cli_official_scanner.py --max-tickers 150 --output-prefix "risk_analysis"

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
python cli_official_scanner.py --max-tickers 100 --log-level DEBUG --output-prefix "validation"

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
python cli_official_scanner.py --max-tickers 500 --output-prefix "market_analysis"

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
python cli_official_scanner.py --max-tickers 10 --log-level DEBUG

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
python cli_official_scanner.py --max-tickers 50

# Use warning level logging
python cli_official_scanner.py --log-level WARNING

# Check system resources
top -l 1 | grep python
```

#### 3. Memory Issues
```bash
# Use smaller batches
python cli_official_scanner.py --max-tickers 25

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
python cli_official_scanner.py  # Uses cache by default

# Check if cache exists
ls -la cache/official_tickers.json
```

### Debug Commands
```bash
# Full debug scan
python cli_official_scanner.py --max-tickers 5 --log-level DEBUG --output-prefix "debug"

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
time python cli_official_scanner.py --max-tickers 100

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
# Generate signals
python cli_official_scanner.py --max-tickers 50 --output-prefix "backtest_signals"

# Extract tickers for backtesting
python -c "
import pandas as pd
df = pd.read_csv('scans/backtest_signals_*.csv')
tickers = df['ticker'].tolist()
print(' '.join(tickers))
" > tickers.txt

# Run backtest (if EVR CLI is available)
# evr backtest --symbols $(cat tickers.txt)
```

### With Trading Platforms
```bash
# Generate signals for trading platform
python cli_official_scanner.py --max-tickers 100 --output-prefix "trading_signals"

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
python cli_official_scanner.py --max-tickers 200 --output-prefix "alerts"

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
# 0 9 * * 1-5 cd /path/to/evr && source .venv/bin/activate && python cli_official_scanner.py --max-tickers 200 --output-prefix "daily_$(date +%Y%m%d)"
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

This usage guide provides practical examples and workflows for using the EVR scanner effectively in various scenarios.

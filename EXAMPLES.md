# EVR Scanner Examples

This document provides practical examples and use cases for the EVR NYSE & NASDAQ Ticker Scanner.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Advanced Examples](#advanced-examples)
3. [Integration Examples](#integration-examples)
4. [Analysis Examples](#analysis-examples)
5. [Automation Examples](#automation-examples)

## Basic Examples

### Example 1: First Scan
```bash
# Run your first scan
python official_scanner.py

# Expected output:
# Found 8034 official tickers
# Scanning 100 tickers...
# Generated 71 signals
# Saved results to scans/signals_20251021_211331.csv
```

### Example 2: Custom Parameters
```bash
# Scan 50 tickers, show top 10 signals
python cli_official_scanner.py --max-tickers 50 --top 10

# Expected output:
# Configuration:
#   Max Tickers: 50
#   Top Signals: 10
#   Output Prefix: signals
#   Log Level: INFO
#   Use Cache: True
```

### Example 3: Debug Mode
```bash
# Enable debug logging for troubleshooting
python cli_official_scanner.py --max-tickers 10 --log-level DEBUG

# Expected output:
# [21:13:57] DEBUG    Generated 2 signals for AAPL
# [21:13:57] DEBUG    AAPL: yfinance received OHLC data: 2024-10-22 13:30:00 -> 2025-10-21 20:00:01
# [21:13:57] DEBUG    Insufficient data for ABLVW
```

## Advanced Examples

### Example 4: Production Scan
```bash
# Large production scan
python cli_official_scanner.py --max-tickers 500 --log-level INFO --output-prefix "production_scan"

# Expected output:
# Configuration:
#   Max Tickers: 500
#   Top Signals: 20
#   Output Prefix: production_scan
#   Log Level: INFO
#   Use Cache: True
# 
# Getting official ticker lists...
# Found 8034 official tickers
# Scanning 500 tickers...
# Generated 312 signals
# Saved results to scans/production_scan_20251021_211500.csv
```

### Example 5: Fresh Data Scan
```bash
# Force fresh ticker data
python cli_official_scanner.py --no-cache --max-tickers 100 --output-prefix "fresh_data"

# Expected output:
# Configuration:
#   Max Tickers: 100
#   Top Signals: 20
#   Output Prefix: fresh_data
#   Log Level: INFO
#   Use Cache: False
# 
# Getting official ticker lists...
# Fetching official ticker lists from NASDAQ FTP...
# Successfully fetched 5142 NASDAQ tickers
# Successfully fetched 2892 NYSE tickers
# Found 8034 official tickers
```

### Example 6: Batch Processing
```bash
# Process in batches
for i in {1..3}; do
    echo "Processing batch $i..."
    python cli_official_scanner.py --max-tickers 100 --output-prefix "batch_$i"
    sleep 30  # Wait 30 seconds between batches
done

# Expected output:
# Processing batch 1...
# Generated 71 signals
# Processing batch 2...
# Generated 68 signals
# Processing batch 3...
# Generated 73 signals
```

## Integration Examples

### Example 7: CSV Analysis
```bash
# Generate signals and analyze
python cli_official_scanner.py --max-tickers 100 --output-prefix "analysis"

# Analyze the results
python -c "
import pandas as pd
import glob

# Find the latest CSV file
files = glob.glob('scans/analysis_*.csv')
if files:
    latest_file = max(files, key=lambda x: x.split('_')[-1])
    df = pd.read_csv(latest_file)
    
    print(f'Analysis of {latest_file}:')
    print(f'Total signals: {len(df)}')
    print(f'Signal types: {df[\"signal_type\"].value_counts().to_dict()}')
    print(f'Directions: {df[\"direction\"].value_counts().to_dict()}')
    print(f'Average confidence: {df[\"confidence\"].mean():.1%}')
    print(f'Average expected return: {df[\"expected_return\"].mean():.2%}')
    print(f'Average risk/reward: {df[\"risk_reward_ratio\"].mean():.2f}')
    
    # Top 5 signals by expected return
    print('\\nTop 5 signals by expected return:')
    top_signals = df.nlargest(5, 'expected_return')
    for _, signal in top_signals.iterrows():
        print(f'{signal[\"ticker\"]}: {signal[\"signal_type\"]} - {signal[\"expected_return\"]:.2%} return')
"
```

### Example 8: JSON Processing
```bash
# Generate signals in JSON format
python cli_official_scanner.py --max-tickers 50 --output-prefix "json_data"

# Process JSON output
python -c "
import json
import glob

# Find the latest JSON file
files = glob.glob('scans/json_data_*.json')
if files:
    latest_file = max(files, key=lambda x: x.split('_')[-1])
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    print(f'JSON Analysis of {latest_file}:')
    print(f'Total signals: {len(data)}')
    
    # Group by signal type
    signal_types = {}
    for signal in data:
        signal_type = signal['signal_type']
        if signal_type not in signal_types:
            signal_types[signal_type] = []
        signal_types[signal_type].append(signal)
    
    print('\\nSignals by type:')
    for signal_type, signals in signal_types.items():
        print(f'{signal_type}: {len(signals)} signals')
        avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
        print(f'  Average confidence: {avg_confidence:.1%}')
    
    # High confidence signals
    high_conf_signals = [s for s in data if s['confidence'] > 0.7]
    print(f'\\nHigh confidence signals (>70%): {len(high_conf_signals)}')
    for signal in high_conf_signals[:5]:
        print(f'{signal[\"ticker\"]}: {signal[\"signal_type\"]} - {signal[\"confidence\"]:.1%} confidence')
"
```

### Example 9: Trading Platform Integration
```bash
# Generate signals for trading platform
python cli_official_scanner.py --max-tickers 100 --output-prefix "trading"

# Convert to trading platform format
python -c "
import pandas as pd
import glob

# Find the latest CSV file
files = glob.glob('scans/trading_*.csv')
if files:
    latest_file = max(files, key=lambda x: x.split('_')[-1])
    df = pd.read_csv(latest_file)
    
    # Filter for high confidence signals
    high_conf = df[df['confidence'] > 0.6]
    
    # Create trading platform format
    trading_format = high_conf[['ticker', 'direction', 'entry_price', 'stop_loss', 'take_profit', 'confidence']].copy()
    trading_format['position_size'] = 1000  # Default position size
    trading_format['notes'] = high_conf['reason']
    
    # Save in trading platform format
    trading_format.to_csv('trading_platform_signals.csv', index=False)
    
    print(f'Exported {len(trading_format)} high-confidence signals to trading_platform_signals.csv')
    print('\\nSample signals:')
    print(trading_format.head().to_string(index=False))
"
```

## Analysis Examples

### Example 10: Market Sentiment Analysis
```bash
# Generate comprehensive market signals
python cli_official_scanner.py --max-tickers 300 --output-prefix "market_sentiment"

# Analyze market sentiment
python -c "
import pandas as pd
import glob

# Find the latest CSV file
files = glob.glob('scans/market_sentiment_*.csv')
if files:
    latest_file = max(files, key=lambda x: x.split('_')[-1])
    df = pd.read_csv(latest_file)
    
    print('Market Sentiment Analysis:')
    print(f'Total signals: {len(df)}')
    
    # Direction analysis
    long_signals = len(df[df['direction'] == 'LONG'])
    short_signals = len(df[df['direction'] == 'SHORT'])
    total_signals = len(df)
    
    print(f'Long signals: {long_signals} ({long_signals/total_signals:.1%})')
    print(f'Short signals: {short_signals} ({short_signals/total_signals:.1%})')
    print(f'Market bias: {\"Bullish\" if long_signals > short_signals else \"Bearish\"}')
    
    # Signal type analysis
    print('\\nSignal type distribution:')
    signal_counts = df['signal_type'].value_counts()
    for signal_type, count in signal_counts.items():
        percentage = count / total_signals * 100
        print(f'{signal_type}: {count} ({percentage:.1f}%)')
    
    # Confidence analysis
    print(f'\\nConfidence analysis:')
    print(f'Average confidence: {df[\"confidence\"].mean():.1%}')
    print(f'High confidence (>70%): {len(df[df[\"confidence\"] > 0.7])} signals')
    print(f'Medium confidence (50-70%): {len(df[(df[\"confidence\"] >= 0.5) & (df[\"confidence\"] <= 0.7)])} signals')
    print(f'Low confidence (<50%): {len(df[df[\"confidence\"] < 0.5])} signals')
    
    # Expected return analysis
    print(f'\\nExpected return analysis:')
    print(f'Average expected return: {df[\"expected_return\"].mean():.2%}')
    print(f'High return (>8%): {len(df[df[\"expected_return\"] > 0.08])} signals')
    print(f'Medium return (5-8%): {len(df[(df[\"expected_return\"] >= 0.05) & (df[\"expected_return\"] <= 0.08)])} signals')
    print(f'Low return (<5%): {len(df[df[\"expected_return\"] < 0.05])} signals')
"
```

### Example 11: Risk Analysis
```bash
# Generate signals for risk analysis
python cli_official_scanner.py --max-tickers 200 --output-prefix "risk_analysis"

# Analyze risk characteristics
python -c "
import pandas as pd
import glob

# Find the latest CSV file
files = glob.glob('scans/risk_analysis_*.csv')
if files:
    latest_file = max(files, key=lambda x: x.split('_')[-1])
    df = pd.read_csv(latest_file)
    
    print('Risk Analysis:')
    print(f'Total signals: {len(df)}')
    
    # Risk-reward analysis
    print(f'\\nRisk-Reward Analysis:')
    print(f'Average R/R ratio: {df[\"risk_reward_ratio\"].mean():.2f}')
    print(f'Min R/R ratio: {df[\"risk_reward_ratio\"].min():.2f}')
    print(f'Max R/R ratio: {df[\"risk_reward_ratio\"].max():.2f}')
    print(f'Signals with R/R > 2.0: {len(df[df[\"risk_reward_ratio\"] > 2.0])}')
    print(f'Signals with R/R > 3.0: {len(df[df[\"risk_reward_ratio\"] > 3.0])}')
    
    # Risk percentage analysis
    print(f'\\nRisk Percentage Analysis:')
    print(f'Average risk: {df[\"risk_percentage\"].mean():.1f}%')
    print(f'High risk (>8%): {len(df[df[\"risk_percentage\"] > 8])} signals')
    print(f'Medium risk (5-8%): {len(df[(df[\"risk_percentage\"] >= 5) & (df[\"risk_percentage\"] <= 8)])} signals')
    print(f'Low risk (<5%): {len(df[df[\"risk_percentage\"] < 5])} signals')
    
    # Reward percentage analysis
    print(f'\\nReward Percentage Analysis:')
    print(f'Average reward: {df[\"reward_percentage\"].mean():.1f}%')
    print(f'High reward (>12%): {len(df[df[\"reward_percentage\"] > 12])} signals')
    print(f'Medium reward (8-12%): {len(df[(df[\"reward_percentage\"] >= 8) & (df[\"reward_percentage\"] <= 12)])} signals')
    print(f'Low reward (<8%): {len(df[df[\"reward_percentage\"] < 8])} signals')
    
    # Best risk-adjusted signals
    print(f'\\nTop 10 Risk-Adjusted Signals (by expected return):')
    top_signals = df.nlargest(10, 'expected_return')
    for _, signal in top_signals.iterrows():
        print(f'{signal[\"ticker\"]}: {signal[\"signal_type\"]} - {signal[\"expected_return\"]:.2%} return, {signal[\"risk_percentage\"]:.1f}% risk')
"
```

### Example 12: Sector Analysis
```bash
# Generate signals for sector analysis
python cli_official_scanner.py --max-tickers 400 --output-prefix "sector_analysis"

# Analyze by sector (requires sector mapping)
python -c "
import pandas as pd
import glob

# Find the latest CSV file
files = glob.glob('scans/sector_analysis_*.csv')
if files:
    latest_file = max(files, key=lambda x: x.split('_')[-1])
    df = pd.read_csv(latest_file)
    
    print('Sector Analysis:')
    print(f'Total signals: {len(df)}')
    
    # Signal type analysis by sector
    print('\\nSignal types by frequency:')
    signal_counts = df['signal_type'].value_counts()
    for signal_type, count in signal_counts.items():
        percentage = count / len(df) * 100
        print(f'{signal_type}: {count} ({percentage:.1f}%)')
    
    # Direction analysis
    print(f'\\nDirection analysis:')
    direction_counts = df['direction'].value_counts()
    for direction, count in direction_counts.items():
        percentage = count / len(df) * 100
        print(f'{direction}: {count} ({percentage:.1f}%)')
    
    # Confidence distribution
    print(f'\\nConfidence distribution:')
    conf_ranges = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    for low, high in conf_ranges:
        count = len(df[(df['confidence'] >= low) & (df['confidence'] < high)])
        percentage = count / len(df) * 100
        print(f'{low:.1f}-{high:.1f}: {count} ({percentage:.1f}%)')
    
    # Expected return distribution
    print(f'\\nExpected return distribution:')
    return_ranges = [(0.0, 0.05), (0.05, 0.08), (0.08, 0.12), (0.12, 0.20), (0.20, 1.0)]
    for low, high in return_ranges:
        count = len(df[(df['expected_return'] >= low) & (df['expected_return'] < high)])
        percentage = count / len(df) * 100
        print(f'{low:.1%}-{high:.1%}: {count} ({percentage:.1f}%)')
"
```

## Automation Examples

### Example 13: Daily Automation Script
```bash
#!/bin/bash
# daily_scanner.sh

cd /path/to/evr
source .venv/bin/activate

# Generate daily signals
echo "Starting daily scan at $(date)"
python cli_official_scanner.py \
    --max-tickers 200 \
    --top 30 \
    --output-prefix "daily_$(date +%Y%m%d)" \
    --log-level INFO

# Check if scan was successful
if [ $? -eq 0 ]; then
    echo "Daily scan completed successfully at $(date)"
    
    # Generate summary email
    echo "Daily Trading Signals Summary" > daily_summary.txt
    echo "Generated: $(date)" >> daily_summary.txt
    echo "" >> daily_summary.txt
    
    # Add scan results
    cat scans/daily_$(date +%Y%m%d)_*_summary.txt >> daily_summary.txt
    
    # Send email (if configured)
    # mail -s "Daily Trading Signals - $(date +%Y-%m-%d)" trader@example.com < daily_summary.txt
    
    echo "Summary generated: daily_summary.txt"
else
    echo "Daily scan failed at $(date)"
    exit 1
fi
```

### Example 14: Weekly Analysis Script
```bash
#!/bin/bash
# weekly_analysis.sh

cd /path/to/evr
source .venv/bin/activate

# Generate weekly comprehensive scan
echo "Starting weekly analysis at $(date)"
python cli_official_scanner.py \
    --max-tickers 500 \
    --top 50 \
    --output-prefix "weekly_$(date +%Y%m%d)" \
    --log-level INFO

# Generate weekly report
python -c "
import pandas as pd
import glob
from datetime import datetime, timedelta

# Find this week's scan
files = glob.glob('scans/weekly_*.csv')
if files:
    latest_file = max(files, key=lambda x: x.split('_')[-1])
    df = pd.read_csv(latest_file)
    
    print('Weekly Analysis Report')
    print('=' * 50)
    print(f'Generated: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
    print(f'Total signals: {len(df)}')
    print()
    
    # Signal type breakdown
    print('Signal Type Breakdown:')
    signal_counts = df['signal_type'].value_counts()
    for signal_type, count in signal_counts.items():
        percentage = count / len(df) * 100
        print(f'  {signal_type}: {count} ({percentage:.1f}%)')
    print()
    
    # Direction breakdown
    print('Direction Breakdown:')
    direction_counts = df['direction'].value_counts()
    for direction, count in direction_counts.items():
        percentage = count / len(df) * 100
        print(f'  {direction}: {count} ({percentage:.1f}%)')
    print()
    
    # Top performers
    print('Top 10 Signals by Expected Return:')
    top_signals = df.nlargest(10, 'expected_return')
    for i, (_, signal) in enumerate(top_signals.iterrows(), 1):
        print(f'  {i:2d}. {signal[\"ticker\"]:6s} {signal[\"signal_type\"]:15s} {signal[\"expected_return\"]:6.2%} ({signal[\"confidence\"]:5.1%})')
    print()
    
    # Risk analysis
    print('Risk Analysis:')
    print(f'  Average risk: {df[\"risk_percentage\"].mean():.1f}%')
    print(f'  Average reward: {df[\"reward_percentage\"].mean():.1f}%')
    print(f'  Average R/R ratio: {df[\"risk_reward_ratio\"].mean():.2f}')
    print()
    
    # Confidence analysis
    print('Confidence Analysis:')
    high_conf = len(df[df['confidence'] > 0.7])
    med_conf = len(df[(df['confidence'] >= 0.5) & (df['confidence'] <= 0.7)])
    low_conf = len(df[df['confidence'] < 0.5])
    print(f'  High confidence (>70%): {high_conf} ({high_conf/len(df)*100:.1f}%)')
    print(f'  Medium confidence (50-70%): {med_conf} ({med_conf/len(df)*100:.1f}%)')
    print(f'  Low confidence (<50%): {low_conf} ({low_conf/len(df)*100:.1f}%)')
" > weekly_report.txt

echo "Weekly analysis completed at $(date)"
echo "Report generated: weekly_report.txt"
```

### Example 15: Monitoring Script
```bash
#!/bin/bash
# monitor_scanner.sh

cd /path/to/evr
source .venv/bin/activate

# Check scanner health
echo "Scanner Health Check at $(date)"
echo "=================================="

# Check if cache exists and is recent
if [ -f "cache/official_tickers.json" ]; then
    cache_age=$(($(date +%s) - $(stat -f %m cache/official_tickers.json)))
    cache_hours=$((cache_age / 3600))
    echo "✓ Cache exists (age: ${cache_hours} hours)"
    
    if [ $cache_hours -gt 24 ]; then
        echo "⚠ Warning: Cache is older than 24 hours"
    fi
else
    echo "✗ Cache file not found"
fi

# Check recent scan results
recent_scans=$(find scans/ -name "*.csv" -mtime -1 | wc -l)
echo "Recent scans (last 24h): $recent_scans"

if [ $recent_scans -eq 0 ]; then
    echo "⚠ Warning: No recent scans found"
fi

# Check disk space
disk_usage=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
echo "Disk usage: ${disk_usage}%"

if [ $disk_usage -gt 80 ]; then
    echo "⚠ Warning: Disk usage is high"
fi

# Test scanner functionality
echo "Testing scanner functionality..."
python cli_official_scanner.py --max-tickers 5 --log-level ERROR --output-prefix "health_check"

if [ $? -eq 0 ]; then
    echo "✓ Scanner test passed"
else
    echo "✗ Scanner test failed"
fi

echo "Health check completed at $(date)"
```

These examples demonstrate various ways to use the EVR scanner for different purposes, from basic usage to advanced automation and analysis.

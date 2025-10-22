# EVR NYSE & NASDAQ Ticker Scanner

A comprehensive trading signal scanner that uses official NASDAQ FTP data sources to scan thousands of NYSE and NASDAQ tickers and generate trading recommendations using technical analysis.

## Features

- **Official Data Sources**: Uses NASDAQ's official FTP feeds for comprehensive ticker lists
- **Comprehensive Coverage**: Scans 8,034+ official NYSE and NASDAQ tickers
- **Technical Analysis**: Multiple signal types including Bollinger Bands, Moving Averages, RSI, MACD
- **Signal Aggregation**: Combines multiple signals per ticker into ranked recommendations (default behavior)
- **Risk Management**: Built-in stop-loss, take-profit, and risk-reward calculations
- **Progress Tracking**: Real-time progress bars and detailed logging
- **Caching**: 24-hour cache for ticker lists to reduce API calls
- **Multiple Outputs**: CSV, JSON, and summary text files

## Installation

### Prerequisites

- Python 3.11+
- uv package manager (recommended) or pip

### Install Dependencies

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install pandas numpy pyarrow yfinance requests rich matplotlib jinja2 seaborn lxml html5lib

# Install EVR framework
uv pip install -e .

# Or using pip
pip install pandas numpy pyarrow yfinance requests rich matplotlib jinja2 seaborn lxml html5lib
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Run the scanner with aggregated recommendations (scans all tickers by default)
python official_scanner.py

# Run with individual signals (disable aggregation)
python official_scanner.py --no-aggregate
```

### CLI Options

```bash
python official_scanner.py --help
```

**Available Options:**
- `--max-tickers`: Maximum number of tickers to scan (default: all tickers)
- `--top`: Number of top recommendations to display (default: 20)
- `--output-prefix`: Prefix for output files (default: signals)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--no-cache`: Force fresh ticker fetch (ignore cache)
- `--no-aggregate`: Disable aggregation and show individual signals

## Usage Examples

### Basic Scan
```bash
# Scan 100 tickers, show top 20 signals
python cli_official_scanner.py
```

### Custom Configuration
```bash
# Scan 200 tickers, show top 30 signals
python cli_official_scanner.py --max-tickers 200 --top 30

# Enable debug logging
python cli_official_scanner.py --log-level DEBUG

# Force fresh ticker data
python cli_official_scanner.py --no-cache

# Custom output prefix
python cli_official_scanner.py --output-prefix "daily_scan"
```

### Production Usage
```bash
# Large scan with debug logging
python cli_official_scanner.py --max-tickers 500 --log-level INFO --output-prefix "production"

# Quick scan for testing
python cli_official_scanner.py --max-tickers 50 --top 10 --output-prefix "test"
```

## Data Sources

### Official NASDAQ FTP Feeds

The scanner uses NASDAQ's official FTP feeds:

- **NASDAQ Tickers**: `ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt`
- **NYSE Tickers**: `ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt`

### Ticker Coverage

- **Total Tickers**: 8,034+ official tickers
- **NASDAQ**: 5,142+ tickers
- **NYSE**: 2,892+ tickers
- **Caching**: 24-hour cache to reduce API calls

## Signal Aggregation

The scanner can aggregate multiple signals per ticker into ranked recommendations using a composite scoring algorithm:

### Aggregation Features
- **Signal Weighting**: Combines signals based on confidence and expected return
- **Consensus Analysis**: Measures agreement between signals on direction
- **Diversity Scoring**: Rewards tickers with multiple signal types
- **Composite Scoring**: Weighted combination of confidence, expected return, diversity, and consensus
- **Ranked Output**: Sorts tickers by composite score for easy prioritization

### Usage
```bash
# Default behavior (aggregation enabled)
python official_scanner.py --max-tickers 200 --top 15

# Disable aggregation for individual signals
python official_scanner.py --no-aggregate --max-tickers 100 --top 20
```

### Enhanced Composite Score Calculation
The composite score combines multiple sophisticated metrics:
- **Probability Score** (25% weight): Based on signal strength and historical effectiveness
- **Payoff Score** (25% weight): Risk-reward ratios and expected returns
- **Risk-Adjusted Return** (20% weight): Sharpe-like ratio with consistency factor
- **Confidence** (15% weight): Average signal confidence
- **Signal Diversity** (10% weight): Number of different signal types
- **Consensus** (5% weight): Agreement on direction

### Output Format
Aggregated results include:
- Ranked ticker recommendations
- Enhanced composite scores for ranking
- Probability scores (signal effectiveness)
- Payoff scores (risk-reward analysis)
- Risk-adjusted return scores
- Signal type breakdown per ticker
- Average confidence and expected return
- Primary direction (LONG/SHORT)

## Signal Types

### 1. Bollinger Band Breakout
- **Trigger**: Price near upper Bollinger Band (position > 0.8)
- **Direction**: LONG
- **Confidence**: 55%
- **Risk/Reward**: 2.5:1
- **Stop Loss**: 6%
- **Take Profit**: 15%

### 2. Moving Average Crossover
- **Trigger**: Price above 20-day SMA, which is above 50-day SMA
- **Direction**: LONG
- **Confidence**: 70%
- **Risk/Reward**: 2.0:1
- **Stop Loss**: 5%
- **Take Profit**: 10%

### 3. RSI Oversold/Overbought
- **Trigger**: RSI < 30 (oversold) or RSI > 70 (overbought)
- **Direction**: LONG (oversold) or SHORT (overbought)
- **Confidence**: 60%
- **Risk/Reward**: 1.0:1
- **Stop Loss**: 8%
- **Take Profit**: 8%

### 4. MACD Bullish
- **Trigger**: MACD line above signal line with positive histogram
- **Direction**: LONG
- **Confidence**: 65%
- **Risk/Reward**: 3.0:1
- **Stop Loss**: 4%
- **Take Profit**: 12%

### 5. Volume Momentum
- **Trigger**: High volume (>1.5x average) with positive momentum (>2%)
- **Direction**: LONG
- **Confidence**: 60%
- **Risk/Reward**: 1.86:1
- **Stop Loss**: 7%
- **Take Profit**: 13%

## Output Files

### CSV Format
```csv
ticker,signal_type,direction,entry_price,stop_loss,take_profit,confidence,reason,risk_reward_ratio,risk_percentage,reward_percentage,expected_return
AAPL,bb_breakout,LONG,262.77,247.00,302.19,0.55,Price near upper Bollinger Band (position: 0.95),2.5,6.0,15.0,0.0825
```

### JSON Format
```json
{
  "ticker": "AAPL",
  "signal_type": "bb_breakout",
  "direction": "LONG",
  "entry_price": 262.77,
  "stop_loss": 247.00,
  "take_profit": 302.19,
  "confidence": 0.55,
  "reason": "Price near upper Bollinger Band (position: 0.95)",
  "risk_reward_ratio": 2.5,
  "expected_return": 0.0825
}
```

### Summary Format
```
EVR Trading Signals Summary
Generated: 2025-10-21 21:13:57
Total Signals: 45

Signals by Type:
  bb_breakout: 25
  ma_crossover: 15
  rsi_overbought: 3
  volume_momentum: 2

Top 10 Signals:
 1. AAPL   bb_breakout     LONG  Conf: 55.0% Expected:  8.25%
 2. MSFT   ma_crossover    LONG  Conf: 70.0% Expected:  7.00%
 3. GOOGL  bb_breakout     LONG  Conf: 55.0% Expected:  8.25%
```

## Performance

### Scanning Speed
- **100 tickers**: ~30 seconds
- **500 tickers**: ~2.5 minutes
- **1000 tickers**: ~5 minutes
- **Rate limiting**: 0.1s between requests

### Signal Quality
- **Average confidence**: 62-63%
- **Average expected return**: 6.99-7.27%
- **Long bias**: 87-89% of signals
- **Risk/reward ratio**: 2.0-2.5:1

### Memory Usage
- **Ticker list**: ~95KB cached
- **Scan results**: ~20-50KB per 100 signals
- **Efficient processing**: Batch processing with cleanup

## Logging

### Log Levels

#### INFO (Default)
```
[21:13:57] INFO     Starting scan of 100 tickers
[21:13:57] INFO     Scan completed: 71 signals from 100 tickers
[21:13:57] INFO     Saved 71 signals to scans/signals_20251021_211331.csv
```

#### DEBUG
```
[21:13:57] DEBUG    Generated 2 signals for AAPL
[21:13:57] DEBUG    AAPL: yfinance received OHLC data: 2024-10-22 13:30:00 -> 2025-10-21 20:00:01
[21:13:57] DEBUG    Insufficient data for ABLVW
```

#### WARNING
```
[21:13:57] WARNING  Error processing AAM.U: No data found, symbol may be delisted
```

#### ERROR
```
[21:13:57] ERROR    Failed to fetch NASDAQ tickers: Connection timeout
```

## Configuration

### Environment Variables
```bash
# Optional: Set custom cache directory
export EVR_CACHE_DIR="~/.evr/cache"

# Optional: Set custom output directory
export EVR_OUTPUT_DIR="./scans"
```

### Cache Management
```bash
# View cached ticker list
cat cache/official_tickers.json | jq '.tickers | length'

# Clear cache
rm cache/official_tickers.json

# Force fresh fetch
python cli_official_scanner.py --no-cache
```

## Troubleshooting

### Common Issues

#### 1. No Signals Generated
**Problem**: Scanner runs but generates no signals
**Solutions**:
- Check if tickers have sufficient data (need 50+ days)
- Try increasing `--max-tickers` to scan more symbols
- Use `--log-level DEBUG` to see detailed processing

#### 2. API Rate Limits
**Problem**: Yahoo Finance API rate limit errors
**Solutions**:
- Reduce `--max-tickers` for smaller batches
- Increase rate limiting delay in code (currently 0.1s)
- Use cached ticker list to avoid repeated fetches

#### 3. Memory Issues
**Problem**: Out of memory errors with large scans
**Solutions**:
- Reduce `--max-tickers` to smaller batches
- Close other applications to free memory
- Use `--no-cache` to avoid loading large ticker lists

#### 4. Network Issues
**Problem**: Failed to fetch ticker data from NASDAQ FTP
**Solutions**:
- Check internet connection
- Try again later (NASDAQ FTP may be temporarily unavailable)
- Use cached ticker list if available

### Debug Mode
```bash
# Enable debug logging for troubleshooting
python cli_official_scanner.py --log-level DEBUG --max-tickers 10
```

### Performance Optimization
```bash
# For faster scanning
python cli_official_scanner.py --max-tickers 50 --log-level WARNING

# For comprehensive scanning
python cli_official_scanner.py --max-tickers 1000 --log-level INFO
```

## Integration

### With EVR Framework
```bash
# Generate signals
python cli_official_scanner.py --max-tickers 100 --output-prefix "daily"

# Backtest signals using EVR CLI
evr backtest --symbols $(cat scans/daily_*.csv | cut -d',' -f1 | tail -n +2)
```

### With External Systems
```bash
# Export to trading platform
python cli_official_scanner.py --output-prefix "trading_signals"
# Import CSV into trading platform

# API integration
python cli_official_scanner.py --output-prefix "api_signals"
# Use JSON output for API integration
```

### Automation
```bash
# Daily scan script
#!/bin/bash
cd /path/to/evr
source .venv/bin/activate
python cli_official_scanner.py --max-tickers 200 --output-prefix "daily_$(date +%Y%m%d)"
```

## Advanced Usage

### Custom Signal Filters
Modify `official_scanner.py` to add custom signal filters:

```python
def generate_signals(self, ticker: str, data: pd.DataFrame, indicators: Dict[str, float]) -> List[Dict[str, Any]]:
    signals = []
    
    # Add custom signal logic here
    if custom_condition:
        signals.append({
            'ticker': ticker,
            'signal_type': 'custom_signal',
            'direction': 'LONG',
            'entry_price': current_price,
            'stop_loss': current_price * 0.95,
            'take_profit': current_price * 1.10,
            'confidence': 0.7,
            'reason': 'Custom signal logic'
        })
    
    return signals
```

### Custom Technical Indicators
Add custom indicators to `calculate_technical_indicators()`:

```python
def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
    indicators = {}
    
    # Add custom indicators
    indicators['custom_indicator'] = calculate_custom_indicator(data)
    
    return indicators
```

## Support

### Getting Help
1. Check this documentation
2. Review error messages and logs
3. Use debug mode for detailed information
4. Check GitHub issues
5. Create a new issue with details

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

### Version 1.0.0
- Initial release
- Official NASDAQ FTP integration
- Multiple signal types
- Comprehensive logging
- CLI interface
- Caching system
- Multiple output formats

#!/usr/bin/env python3
"""
Official NYSE/NASDAQ ticker scanner using NASDAQ's official data sources.

This scanner uses the official NASDAQ FTP feeds to get comprehensive
lists of all NYSE and NASDAQ tickers with proper logging and progress tracking.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler


def get_us_tickers(exchange: str) -> pd.DataFrame:
    """Get official US ticker list from NASDAQ FTP.
    
    Args:
        exchange: 'NASDAQ' or 'NYSE'
        
    Returns:
        DataFrame with symbol and exchange columns
        
    Raises:
        ValueError: If exchange is not 'NASDAQ' or 'NYSE'
        ConnectionError: If unable to fetch data from NASDAQ FTP
    """
    logger = logging.getLogger(__name__)
    
    try:
        if exchange.upper() == "NASDAQ":
            logger.info("Fetching NASDAQ tickers from official FTP...")
            url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
            df = pd.read_csv(url, sep="|")
            df = df[df['Symbol'].str.len() > 0]
            result = pd.DataFrame({"symbol": df["Symbol"].str.strip(), "exchange": exchange})
            logger.info(f"Successfully fetched {len(result)} NASDAQ tickers")
            return result
            
        elif exchange.upper() == "NYSE":
            logger.info("Fetching NYSE tickers from official FTP...")
            url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
            df = pd.read_csv(url, sep="|")
            df = df[df["Exchange"] == "N"]
            result = pd.DataFrame({"symbol": df["ACT Symbol"].str.strip(), "exchange": exchange})
            logger.info(f"Successfully fetched {len(result)} NYSE tickers")
            return result
            
        else:
            raise ValueError("exchange must be 'NASDAQ' or 'NYSE'")
            
    except Exception as e:
        logger.error(f"Failed to fetch {exchange} tickers: {e}")
        raise ConnectionError(f"Unable to fetch {exchange} tickers from NASDAQ FTP: {e}")


class OfficialTickerScanner:
    """Official ticker scanner using NASDAQ's data sources."""
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize the scanner with logging.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.console = Console()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)]
        )
        self.logger = logging.getLogger(__name__)
        
        # Output directory
        self.output_dir = Path("scans")
        self.output_dir.mkdir(exist_ok=True)
        
        # Cache directory
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.logger.info("OfficialTickerScanner initialized")
    
    def get_comprehensive_tickers(self, use_cache: bool = True) -> List[str]:
        """Get comprehensive list of NYSE and NASDAQ tickers.
        
        Args:
            use_cache: Whether to use cached ticker list if available
            
        Returns:
            List of ticker symbols
        """
        cache_file = self.cache_dir / "official_tickers.json"
        
        # Check cache first
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    cache_time = datetime.fromisoformat(cached_data['timestamp'])
                    # Use cache if less than 24 hours old
                    if datetime.now() - cache_time < timedelta(hours=24):
                        self.logger.info(f"Using cached ticker list ({len(cached_data['tickers'])} tickers)")
                        return cached_data['tickers']
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        
        self.logger.info("Fetching official ticker lists from NASDAQ FTP...")
        
        all_tickers = set()
        
        # Fetch NASDAQ tickers
        try:
            nasdaq_df = get_us_tickers("NASDAQ")
            nasdaq_tickers = nasdaq_df['symbol'].tolist()
            all_tickers.update(nasdaq_tickers)
            self.logger.info(f"Added {len(nasdaq_tickers)} NASDAQ tickers")
        except Exception as e:
            self.logger.error(f"Failed to fetch NASDAQ tickers: {e}")
        
        # Fetch NYSE tickers
        try:
            nyse_df = get_us_tickers("NYSE")
            nyse_tickers = nyse_df['symbol'].tolist()
            all_tickers.update(nyse_tickers)
            self.logger.info(f"Added {len(nyse_tickers)} NYSE tickers")
        except Exception as e:
            self.logger.error(f"Failed to fetch NYSE tickers: {e}")
        
        # Convert to sorted list
        ticker_list = sorted(list(all_tickers))
        
        # Cache the results
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'tickers': ticker_list,
                'nasdaq_count': len(nasdaq_tickers) if 'nasdaq_tickers' in locals() else 0,
                'nyse_count': len(nyse_tickers) if 'nyse_tickers' in locals() else 0
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            self.logger.info(f"Cached {len(ticker_list)} tickers")
        except Exception as e:
            self.logger.warning(f"Failed to cache tickers: {e}")
        
        return ticker_list
    
    def get_stock_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get stock data using yfinance.
        
        Args:
            ticker: Stock symbol
            period: Data period (1y, 6mo, 3mo, etc.)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            return data if not data.empty else None
        except Exception as e:
            self.logger.debug(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators.
        
        Args:
            data: Stock price data
            
        Returns:
            Dictionary of calculated indicators
        """
        if len(data) < 20:
            return {}
        
        indicators = {}
        
        try:
            # Price data
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            # Moving averages
            indicators['sma_20'] = close.rolling(20).mean().iloc[-1]
            indicators['sma_50'] = close.rolling(50).mean().iloc[-1]
            indicators['ema_12'] = close.ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = close.ewm(span=26).mean().iloc[-1]
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            macd_line = indicators['ema_12'] - indicators['ema_26']
            signal_line = pd.Series([macd_line]).ewm(span=9).mean().iloc[-1]
            indicators['macd'] = macd_line
            indicators['macd_signal'] = signal_line
            indicators['macd_histogram'] = macd_line - signal_line
            
            # Bollinger Bands
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            indicators['bb_upper'] = bb_upper.iloc[-1]
            indicators['bb_lower'] = bb_lower.iloc[-1]
            indicators['bb_position'] = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # Volume indicators
            indicators['volume_sma'] = volume.rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_sma']
            
            # Price momentum
            indicators['momentum_5'] = (close.iloc[-1] / close.iloc[-6] - 1) * 100
            indicators['momentum_10'] = (close.iloc[-1] / close.iloc[-11] - 1) * 100
            
            # Volatility
            returns = close.pct_change()
            indicators['volatility_20'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            
        except Exception as e:
            self.logger.debug(f"Error calculating indicators: {e}")
            return {}
        
        return indicators
    
    def generate_signals(self, ticker: str, data: pd.DataFrame, indicators: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate trading signals based on technical indicators.
        
        Args:
            ticker: Stock symbol
            data: Stock price data
            indicators: Calculated technical indicators
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not indicators:
            return signals
        
        current_price = data['Close'].iloc[-1]
        
        # Signal 1: Moving Average Crossover
        if indicators.get('sma_20', 0) > indicators.get('sma_50', 0):
            if current_price > indicators['sma_20']:
                signals.append({
                    'ticker': ticker,
                    'signal_type': 'ma_crossover',
                    'direction': 'LONG',
                    'entry_price': current_price,
                    'stop_loss': current_price * 0.95,  # 5% stop loss
                    'take_profit': current_price * 1.10,  # 10% take profit
                    'confidence': 0.7,
                    'reason': 'Price above 20-day SMA, which is above 50-day SMA'
                })
        
        # Signal 2: RSI Oversold/Overbought
        rsi = indicators.get('rsi', 50)
        if rsi < 30:  # Oversold
            signals.append({
                'ticker': ticker,
                'signal_type': 'rsi_oversold',
                'direction': 'LONG',
                'entry_price': current_price,
                'stop_loss': current_price * 0.92,  # 8% stop loss
                'take_profit': current_price * 1.08,  # 8% take profit
                'confidence': 0.6,
                'reason': f'RSI oversold at {rsi:.1f}'
            })
        elif rsi > 70:  # Overbought
            signals.append({
                'ticker': ticker,
                'signal_type': 'rsi_overbought',
                'direction': 'SHORT',
                'entry_price': current_price,
                'stop_loss': current_price * 1.08,  # 8% stop loss
                'take_profit': current_price * 0.92,  # 8% take profit
                'confidence': 0.6,
                'reason': f'RSI overbought at {rsi:.1f}'
            })
        
        # Signal 3: MACD Signal
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal and indicators.get('macd_histogram', 0) > 0:
            signals.append({
                'ticker': ticker,
                'signal_type': 'macd_bullish',
                'direction': 'LONG',
                'entry_price': current_price,
                'stop_loss': current_price * 0.96,  # 4% stop loss
                'take_profit': current_price * 1.12,  # 12% take profit
                'confidence': 0.65,
                'reason': 'MACD bullish crossover'
            })
        
        # Signal 4: Bollinger Band Breakout
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position > 0.8:  # Near upper band
            signals.append({
                'ticker': ticker,
                'signal_type': 'bb_breakout',
                'direction': 'LONG',
                'entry_price': current_price,
                'stop_loss': current_price * 0.94,  # 6% stop loss
                'take_profit': current_price * 1.15,  # 15% take profit
                'confidence': 0.55,
                'reason': f'Price near upper Bollinger Band (position: {bb_position:.2f})'
            })
        
        # Signal 5: Volume Confirmation
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:  # High volume
            momentum_5 = indicators.get('momentum_5', 0)
            if momentum_5 > 2:  # Positive momentum
                signals.append({
                    'ticker': ticker,
                    'signal_type': 'volume_momentum',
                    'direction': 'LONG',
                    'entry_price': current_price,
                    'stop_loss': current_price * 0.93,  # 7% stop loss
                    'take_profit': current_price * 1.13,  # 13% take profit
                    'confidence': 0.6,
                    'reason': f'High volume ({volume_ratio:.1f}x) with positive momentum ({momentum_5:.1f}%)'
                })
        
        return signals
    
    def scan_tickers(self, tickers: List[str], max_tickers: int = 100) -> List[Dict[str, Any]]:
        """Scan tickers for trading signals.
        
        Args:
            tickers: List of ticker symbols
            max_tickers: Maximum number of tickers to scan
            
        Returns:
            List of trading signals
        """
        self.logger.info(f"Starting scan of {min(len(tickers), max_tickers)} tickers")
        
        all_signals = []
        processed_tickers = tickers[:max_tickers]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Scanning tickers...", total=len(processed_tickers))
            
            for i, ticker in enumerate(processed_tickers):
                try:
                    # Get stock data
                    data = self.get_stock_data(ticker)
                    if data is None or len(data) < 50:
                        self.logger.debug(f"Insufficient data for {ticker}")
                        progress.advance(task)
                        continue
                    
                    # Calculate indicators
                    indicators = self.calculate_technical_indicators(data)
                    if not indicators:
                        self.logger.debug(f"Failed to calculate indicators for {ticker}")
                        progress.advance(task)
                        continue
                    
                    # Generate signals
                    signals = self.generate_signals(ticker, data, indicators)
                    all_signals.extend(signals)
                    
                    if signals:
                        self.logger.debug(f"Generated {len(signals)} signals for {ticker}")
                    
                    progress.advance(task)
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    self.logger.warning(f"Error processing {ticker}: {e}")
                    progress.advance(task)
                    continue
        
        # Sort by confidence and calculate additional metrics
        for signal in all_signals:
            # Calculate risk-reward ratio
            if signal['direction'] == 'LONG':
                risk = signal['entry_price'] - signal['stop_loss']
                reward = signal['take_profit'] - signal['entry_price']
            else:
                risk = signal['stop_loss'] - signal['entry_price']
                reward = signal['entry_price'] - signal['take_profit']
            
            signal['risk_reward_ratio'] = reward / risk if risk > 0 else 0
            signal['risk_percentage'] = abs(risk / signal['entry_price']) * 100
            signal['reward_percentage'] = abs(reward / signal['entry_price']) * 100
            
            # Calculate expected return (simplified)
            signal['expected_return'] = signal['confidence'] * signal['reward_percentage'] / 100
        
        # Sort by expected return
        all_signals.sort(key=lambda x: x['expected_return'], reverse=True)
        
        self.logger.info(f"Scan completed: {len(all_signals)} signals from {len(processed_tickers)} tickers")
        return all_signals
    
    def display_results(self, signals: List[Dict[str, Any]], top_n: int = 20) -> None:
        """Display results in a formatted table.
        
        Args:
            signals: List of trading signals
            top_n: Number of top signals to display
        """
        if not signals:
            self.console.print("[red]No trading signals found[/red]")
            return
        
        # Create table
        table = Table(title=f"Top {min(top_n, len(signals))} Trading Signals")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Ticker", style="magenta", width=8)
        table.add_column("Signal", style="blue", width=15)
        table.add_column("Direction", style="green", width=8)
        table.add_column("Entry Price", style="yellow", width=10)
        table.add_column("Stop Loss", style="red", width=10)
        table.add_column("Take Profit", style="green", width=12)
        table.add_column("Confidence", style="cyan", width=10)
        table.add_column("Expected Return", style="green", width=12)
        table.add_column("Risk/Reward", style="blue", width=10)
        
        for i, signal in enumerate(signals[:top_n], 1):
            table.add_row(
                str(i),
                signal['ticker'],
                signal['signal_type'],
                signal['direction'],
                f"${signal['entry_price']:.2f}",
                f"${signal['stop_loss']:.2f}",
                f"${signal['take_profit']:.2f}",
                f"{signal['confidence']:.1%}",
                f"{signal['expected_return']:.2%}",
                f"{signal['risk_reward_ratio']:.1f}",
            )
        
        self.console.print(table)
        
        # Display summary statistics
        total_signals = len(signals)
        long_signals = len([s for s in signals if s['direction'] == 'LONG'])
        short_signals = len([s for s in signals if s['direction'] == 'SHORT'])
        
        avg_confidence = sum(s['confidence'] for s in signals) / total_signals
        avg_expected_return = sum(s['expected_return'] for s in signals) / total_signals
        
        summary_text = f"""
Summary Statistics:
  Total Signals: {total_signals}
  Long Positions: {long_signals} ({long_signals/total_signals:.1%})
  Short Positions: {short_signals} ({short_signals/total_signals:.1%})
  Average Confidence: {avg_confidence:.1%}
  Average Expected Return: {avg_expected_return:.2%}
        """
        
        panel = Panel(summary_text, title="Scan Results", border_style="green")
        self.console.print(panel)
    
    def save_results(self, signals: List[Dict[str, Any]], filename_prefix: str = "signals") -> None:
        """Save results to files.
        
        Args:
            signals: List of trading signals
            filename_prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_file = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
        df = pd.DataFrame(signals)
        df.to_csv(csv_file, index=False)
        self.logger.info(f"Saved {len(signals)} signals to {csv_file}")
        
        # Save as JSON
        json_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(signals, f, indent=2, default=str)
        self.logger.info(f"Saved detailed results to {json_file}")
        
        # Save summary
        summary_file = self.output_dir / f"{filename_prefix}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"EVR Trading Signals Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Signals: {len(signals)}\n\n")
            
            # Group by signal type
            signal_counts = {}
            for signal in signals:
                signal_type = signal['signal_type']
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            
            f.write("Signals by Type:\n")
            for signal_type, count in signal_counts.items():
                f.write(f"  {signal_type}: {count}\n")
            
            f.write(f"\nTop 10 Signals:\n")
            for i, signal in enumerate(signals[:10], 1):
                f.write(f"{i:2d}. {signal['ticker']:6s} {signal['signal_type']:15s} {signal['direction']:5s} "
                       f"Conf: {signal['confidence']:5.1%} Expected: {signal['expected_return']:6.2%}\n")
        
        self.logger.info(f"Saved summary to {summary_file}")


def main():
    """Main function to run the official scanner."""
    console = Console()
    
    # Display header
    header_text = """
EVR Official Ticker Scanner

This scanner uses NASDAQ's official FTP feeds to get comprehensive
lists of all NYSE and NASDAQ tickers with:
1. Official ticker data from NASDAQ FTP
2. Comprehensive technical analysis
3. Multiple trading signal generation
4. Risk-reward analysis
5. Detailed logging and progress tracking
    """
    
    panel = Panel(header_text, title="EVR Official Scanner", border_style="blue")
    console.print(panel)
    
    try:
        # Initialize scanner with logging
        scanner = OfficialTickerScanner(log_level="INFO")
        
        # Get ticker list
        console.print("\n[blue]Getting official ticker lists...[/blue]")
        tickers = scanner.get_comprehensive_tickers()
        console.print(f"[green]Found {len(tickers)} official tickers[/green]")
        
        # Scan for signals
        console.print("\n[blue]Scanning for trading signals...[/blue]")
        signals = scanner.scan_tickers(tickers, max_tickers=100)  # Limit for demo
        
        # Display results
        console.print("\n[blue]Displaying results...[/blue]")
        scanner.display_results(signals, top_n=20)
        
        # Save results
        console.print("\n[blue]Saving results...[/blue]")
        scanner.save_results(signals)
        
        console.print("\n[green]✅ Scan completed successfully![/green]")
        
        return True
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Scan interrupted by user[/yellow]")
        return False
    except Exception as e:
        console.print(f"\n[red]❌ Error during scan: {e}[/red]")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

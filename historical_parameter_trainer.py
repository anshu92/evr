#!/usr/bin/env python3
"""
Historical Parameter Trainer for EVR Scanner

This module:
1. Fetches historical price data for multiple tickers
2. Simulates trading signals and trades
3. Calculates empirical p_win, avg_win, avg_loss, and other parameters
4. Generates priors and weights for the scanner's probability model
5. Exports trained parameters for integration with the scanner
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import pickle

import pandas as pd
import numpy as np
import yfinance as yf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class TradeResult:
    """Result of a simulated trade."""
    ticker: str
    setup: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    direction: int  # 1 for long, -1 for short
    return_pct: float
    r_multiple: float  # Return in R units
    is_win: bool
    stop_price: float
    target_price: float
    exit_reason: str  # "TARGET", "STOP", "TIME"


@dataclass
class SetupStatistics:
    """Statistics for a trading setup."""
    setup: str
    ticker: str = "ALL"  # "ALL" for global stats, or specific ticker
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    p_win: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_r_win: float = 0.0
    avg_r_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    # Beta prior parameters
    alpha: float = 1.0
    beta: float = 1.0


class TechnicalIndicators:
    """Calculate technical indicators for signal generation."""
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data['Close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data['Close'].rolling(window=period).mean()


class SignalGenerator:
    """Generate trading signals from historical data."""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame, ticker: str) -> List[Dict[str, Any]]:
        """Generate trading signals from price data.
        
        Returns list of signals with:
        - setup: name of the setup
        - direction: 1 for long, -1 for short
        - date: signal date
        - entry_price: entry price
        - stop_price: stop loss price
        - target_price: take profit price
        """
        signals = []
        
        # Calculate indicators
        data = data.copy()
        data['RSI'] = self.indicators.calculate_rsi(data)
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = self.indicators.calculate_macd(data)
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = self.indicators.calculate_bollinger_bands(data)
        data['ATR'] = self.indicators.calculate_atr(data)
        data['EMA_20'] = self.indicators.calculate_ema(data, 20)
        data['EMA_50'] = self.indicators.calculate_ema(data, 50)
        data['SMA_200'] = self.indicators.calculate_sma(data, 200)
        
        # Skip first 200 days to ensure indicators are warmed up
        start_idx = 200
        
        for i in range(start_idx, len(data) - 1):  # -1 to ensure we have at least one future bar
            row = data.iloc[i]
            
            # Skip if essential data is missing
            if pd.isna(row['RSI']) or pd.isna(row['ATR']) or row['ATR'] == 0:
                continue
            
            # Setup 1: RSI Oversold Bounce (Long)
            if row['RSI'] < 30 and row['Close'] > row['EMA_20']:
                atr_multiple = 2.0
                signals.append({
                    'ticker': ticker,
                    'setup': 'RSI_Oversold_Long',
                    'direction': 1,
                    'date': data.index[i],
                    'entry_price': row['Close'],
                    'stop_price': row['Close'] - atr_multiple * row['ATR'],
                    'target_price': row['Close'] + 2 * atr_multiple * row['ATR'],
                })
            
            # Setup 2: RSI Overbought Reversal (Short)
            if row['RSI'] > 70 and row['Close'] < row['EMA_20']:
                atr_multiple = 2.0
                signals.append({
                    'ticker': ticker,
                    'setup': 'RSI_Overbought_Short',
                    'direction': -1,
                    'date': data.index[i],
                    'entry_price': row['Close'],
                    'stop_price': row['Close'] + atr_multiple * row['ATR'],
                    'target_price': row['Close'] - 2 * atr_multiple * row['ATR'],
                })
            
            # Setup 3: MACD Crossover (Long)
            if i > 0:
                prev_row = data.iloc[i-1]
                if (row['MACD'] > row['MACD_Signal'] and 
                    prev_row['MACD'] <= prev_row['MACD_Signal'] and
                    row['Close'] > row['EMA_50']):
                    atr_multiple = 2.0
                    signals.append({
                        'ticker': ticker,
                        'setup': 'MACD_Cross_Long',
                        'direction': 1,
                        'date': data.index[i],
                        'entry_price': row['Close'],
                        'stop_price': row['Close'] - atr_multiple * row['ATR'],
                        'target_price': row['Close'] + 2 * atr_multiple * row['ATR'],
                    })
            
            # Setup 4: Bollinger Band Bounce (Long)
            if row['Close'] <= row['BB_Lower'] and row['Close'] > row['SMA_200']:
                atr_multiple = 2.0
                signals.append({
                    'ticker': ticker,
                    'setup': 'BB_Bounce_Long',
                    'direction': 1,
                    'date': data.index[i],
                    'entry_price': row['Close'],
                    'stop_price': row['Close'] - atr_multiple * row['ATR'],
                    'target_price': row['BB_Middle'],
                })
            
            # Setup 5: Trend Following (Long)
            if (row['EMA_20'] > row['EMA_50'] > row['SMA_200'] and
                row['Close'] > row['EMA_20'] and
                row['RSI'] > 50 and row['RSI'] < 70):
                atr_multiple = 2.5
                signals.append({
                    'ticker': ticker,
                    'setup': 'Trend_Following_Long',
                    'direction': 1,
                    'date': data.index[i],
                    'entry_price': row['Close'],
                    'stop_price': row['Close'] - atr_multiple * row['ATR'],
                    'target_price': row['Close'] + 3 * atr_multiple * row['ATR'],
                })
            
            # Setup 6: Mean Reversion (Short)
            if row['Close'] >= row['BB_Upper'] and row['RSI'] > 70:
                atr_multiple = 2.0
                signals.append({
                    'ticker': ticker,
                    'setup': 'Mean_Reversion_Short',
                    'direction': -1,
                    'date': data.index[i],
                    'entry_price': row['Close'],
                    'stop_price': row['Close'] + atr_multiple * row['ATR'],
                    'target_price': row['BB_Middle'],
                })
        
        return signals


class TradeSimulator:
    """Simulate trades based on signals and calculate outcomes."""
    
    def __init__(self, max_holding_period: int = 20):
        """Initialize trade simulator.
        
        Args:
            max_holding_period: Maximum days to hold a position before closing
        """
        self.max_holding_period = max_holding_period
    
    def simulate_trade(self, signal: Dict[str, Any], future_data: pd.DataFrame) -> Optional[TradeResult]:
        """Simulate a trade from signal to exit.
        
        Args:
            signal: Signal dictionary with entry, stop, target
            future_data: Future price data after signal date
            
        Returns:
            TradeResult object or None if simulation fails
        """
        if future_data.empty:
            return None
        
        entry_price = signal['entry_price']
        stop_price = signal['stop_price']
        target_price = signal['target_price']
        direction = signal['direction']
        
        # Track the trade day by day
        for i, (date, row) in enumerate(future_data.iterrows()):
            if i >= self.max_holding_period:
                # Time stop - exit at close
                exit_price = row['Close']
                exit_reason = "TIME"
                exit_date = date
                break
            
            # Check for stop or target hit
            if direction == 1:  # Long
                # Check if stop was hit
                if row['Low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = "STOP"
                    exit_date = date
                    break
                # Check if target was hit
                elif row['High'] >= target_price:
                    exit_price = target_price
                    exit_reason = "TARGET"
                    exit_date = date
                    break
            else:  # Short (direction == -1)
                # Check if stop was hit (price went up)
                if row['High'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = "STOP"
                    exit_date = date
                    break
                # Check if target was hit (price went down)
                elif row['Low'] <= target_price:
                    exit_price = target_price
                    exit_reason = "TARGET"
                    exit_date = date
                    break
        else:
            # Reached end of data without stop/target
            exit_price = future_data.iloc[-1]['Close']
            exit_reason = "TIME"
            exit_date = future_data.index[-1]
        
        # Calculate returns
        if direction == 1:  # Long
            return_pct = (exit_price - entry_price) / entry_price
            r_unit = abs(entry_price - stop_price)
            r_multiple = (exit_price - entry_price) / r_unit if r_unit > 0 else 0
        else:  # Short
            return_pct = (entry_price - exit_price) / entry_price
            r_unit = abs(stop_price - entry_price)
            r_multiple = (entry_price - exit_price) / r_unit if r_unit > 0 else 0
        
        is_win = return_pct > 0
        
        return TradeResult(
            ticker=signal['ticker'],
            setup=signal['setup'],
            entry_date=signal['date'],
            entry_price=entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            direction=direction,
            return_pct=return_pct,
            r_multiple=r_multiple,
            is_win=is_win,
            stop_price=stop_price,
            target_price=target_price,
            exit_reason=exit_reason
        )


class ParameterEstimator:
    """Estimate parameters from trade results."""
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """Initialize parameter estimator.
        
        Args:
            alpha_prior: Beta distribution alpha prior
            beta_prior: Beta distribution beta prior
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
    
    def calculate_statistics(self, trades: List[TradeResult], 
                            by_setup: bool = True,
                            by_ticker: bool = False) -> Dict[str, SetupStatistics]:
        """Calculate statistics from trade results.
        
        Args:
            trades: List of trade results
            by_setup: Group by setup
            by_ticker: Group by ticker (within setup)
            
        Returns:
            Dictionary of statistics keyed by (setup, ticker) or just setup
        """
        stats_dict = {}
        
        if by_setup and by_ticker:
            # Group by both setup and ticker
            grouped = defaultdict(list)
            for trade in trades:
                key = (trade.setup, trade.ticker)
                grouped[key].append(trade)
        elif by_setup:
            # Group by setup only
            grouped = defaultdict(list)
            for trade in trades:
                key = (trade.setup, "ALL")
                grouped[key].append(trade)
        else:
            # All trades together
            grouped = {("ALL", "ALL"): trades}
        
        # Calculate statistics for each group
        for key, trade_list in grouped.items():
            setup, ticker = key
            stats = self._calculate_group_stats(trade_list, setup, ticker)
            stats_dict[key] = stats
        
        return stats_dict
    
    def _calculate_group_stats(self, trades: List[TradeResult], 
                               setup: str, ticker: str) -> SetupStatistics:
        """Calculate statistics for a group of trades."""
        if not trades:
            return SetupStatistics(setup=setup, ticker=ticker)
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.is_win)
        losing_trades = total_trades - winning_trades
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Bayesian estimate of p_win
        p_win = (winning_trades + self.alpha_prior) / (total_trades + self.alpha_prior + self.beta_prior)
        
        # Average returns
        winning_returns = [t.return_pct for t in trades if t.is_win]
        losing_returns = [t.return_pct for t in trades if not t.is_win]
        
        avg_win = np.mean(winning_returns) if winning_returns else 0.0
        avg_loss = np.mean(losing_returns) if losing_returns else 0.0
        
        # R-multiple statistics
        winning_r = [t.r_multiple for t in trades if t.is_win]
        losing_r = [t.r_multiple for t in trades if not t.is_win]
        
        avg_r_win = np.mean(winning_r) if winning_r else 0.0
        avg_r_loss = np.mean(losing_r) if losing_r else 0.0
        
        # Max win/loss
        max_win = max(winning_returns) if winning_returns else 0.0
        max_loss = min(losing_returns) if losing_returns else 0.0
        
        # Expectancy
        expectancy = p_win * avg_win + (1 - p_win) * avg_loss
        
        # Profit factor
        total_wins = sum(winning_returns) if winning_returns else 0.0
        total_losses = abs(sum(losing_returns)) if losing_returns else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # Beta parameters for the posterior
        alpha = winning_trades + self.alpha_prior
        beta = losing_trades + self.beta_prior
        
        return SetupStatistics(
            setup=setup,
            ticker=ticker,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            p_win=p_win,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_r_win=avg_r_win,
            avg_r_loss=avg_r_loss,
            max_win=max_win,
            max_loss=max_loss,
            expectancy=expectancy,
            profit_factor=profit_factor,
            win_rate=win_rate,
            alpha=alpha,
            beta=beta
        )


class HistoricalParameterTrainer:
    """Main class for training parameters from historical data."""
    
    def __init__(self, tickers: List[str], lookback_months: int = 12,
                 output_dir: str = "trained_parameters"):
        """Initialize trainer.
        
        Args:
            tickers: List of ticker symbols to train on
            lookback_months: Number of months of historical data
            output_dir: Directory to save trained parameters
        """
        self.tickers = tickers
        self.lookback_months = lookback_months
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.signal_generator = SignalGenerator()
        self.trade_simulator = TradeSimulator()
        self.parameter_estimator = ParameterEstimator()
        
        self.historical_data = {}
        self.all_signals = []
        self.all_trades = []
        self.statistics = {}
    
    def fetch_historical_data(self) -> None:
        """Fetch historical data for all tickers."""
        console.print(f"\n[bold cyan]Fetching historical data for {len(self.tickers)} tickers...[/bold cyan]")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * self.lookback_months)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Fetching data...", total=len(self.tickers))
            
            for ticker in self.tickers:
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.history(start=start_date, end=end_date)
                    
                    if not data.empty and len(data) > 200:  # Need sufficient data for indicators
                        self.historical_data[ticker] = data
                        logger.info(f"Fetched {len(data)} bars for {ticker}")
                    else:
                        logger.warning(f"Insufficient data for {ticker}")
                except Exception as e:
                    logger.error(f"Error fetching {ticker}: {e}")
                
                progress.advance(task)
        
        console.print(f"[green]Successfully fetched data for {len(self.historical_data)} tickers[/green]")
    
    def generate_all_signals(self) -> None:
        """Generate trading signals from all historical data."""
        console.print(f"\n[bold cyan]Generating trading signals...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Generating signals...", total=len(self.historical_data))
            
            for ticker, data in self.historical_data.items():
                signals = self.signal_generator.generate_signals(data, ticker)
                self.all_signals.extend(signals)
                logger.info(f"Generated {len(signals)} signals for {ticker}")
                progress.advance(task)
        
        console.print(f"[green]Generated {len(self.all_signals)} total signals[/green]")
    
    def simulate_all_trades(self) -> None:
        """Simulate trades from all signals."""
        console.print(f"\n[bold cyan]Simulating trades...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Simulating trades...", total=len(self.all_signals))
            
            for signal in self.all_signals:
                ticker = signal['ticker']
                signal_date = signal['date']
                
                # Get future data after signal
                data = self.historical_data[ticker]
                future_data = data[data.index > signal_date]
                
                # Simulate trade
                trade_result = self.trade_simulator.simulate_trade(signal, future_data)
                
                if trade_result:
                    self.all_trades.append(trade_result)
                
                progress.advance(task)
        
        console.print(f"[green]Simulated {len(self.all_trades)} trades[/green]")
    
    def calculate_statistics(self) -> None:
        """Calculate statistics from simulated trades."""
        console.print(f"\n[bold cyan]Calculating statistics...[/bold cyan]")
        
        # Overall statistics
        self.statistics['overall'] = self.parameter_estimator.calculate_statistics(
            self.all_trades, by_setup=False, by_ticker=False
        )
        
        # By setup
        self.statistics['by_setup'] = self.parameter_estimator.calculate_statistics(
            self.all_trades, by_setup=True, by_ticker=False
        )
        
        # By setup and ticker
        self.statistics['by_setup_ticker'] = self.parameter_estimator.calculate_statistics(
            self.all_trades, by_setup=True, by_ticker=True
        )
        
        console.print("[green]Statistics calculated[/green]")
    
    def display_results(self) -> None:
        """Display training results in a rich table."""
        console.print("\n" + "="*80)
        console.print("[bold cyan]TRAINING RESULTS[/bold cyan]")
        console.print("="*80 + "\n")
        
        # Overall statistics
        overall_stats = list(self.statistics['overall'].values())[0]
        
        overall_table = Table(title="Overall Statistics", show_header=True)
        overall_table.add_column("Metric", style="cyan")
        overall_table.add_column("Value", style="green")
        
        overall_table.add_row("Total Trades", str(overall_stats.total_trades))
        overall_table.add_row("Winning Trades", str(overall_stats.winning_trades))
        overall_table.add_row("Losing Trades", str(overall_stats.losing_trades))
        overall_table.add_row("Win Rate", f"{overall_stats.win_rate:.2%}")
        overall_table.add_row("P(Win) Bayesian", f"{overall_stats.p_win:.2%}")
        overall_table.add_row("Avg Win", f"{overall_stats.avg_win:.2%}")
        overall_table.add_row("Avg Loss", f"{overall_stats.avg_loss:.2%}")
        overall_table.add_row("Expectancy", f"{overall_stats.expectancy:.2%}")
        overall_table.add_row("Profit Factor", f"{overall_stats.profit_factor:.2f}")
        
        console.print(overall_table)
        console.print()
        
        # By setup statistics
        setup_table = Table(title="Statistics by Setup", show_header=True)
        setup_table.add_column("Setup", style="cyan")
        setup_table.add_column("Trades", justify="right")
        setup_table.add_column("Win Rate", justify="right")
        setup_table.add_column("P(Win)", justify="right")
        setup_table.add_column("Avg Win", justify="right")
        setup_table.add_column("Avg Loss", justify="right")
        setup_table.add_column("Expectancy", justify="right")
        setup_table.add_column("Profit Factor", justify="right")
        
        for (setup, ticker), stats in self.statistics['by_setup'].items():
            setup_table.add_row(
                setup,
                str(stats.total_trades),
                f"{stats.win_rate:.1%}",
                f"{stats.p_win:.1%}",
                f"{stats.avg_win:.1%}",
                f"{stats.avg_loss:.1%}",
                f"{stats.expectancy:.2%}",
                f"{stats.profit_factor:.2f}"
            )
        
        console.print(setup_table)
    
    def save_parameters(self) -> None:
        """Save trained parameters to files."""
        console.print(f"\n[bold cyan]Saving parameters...[/bold cyan]")
        
        # Save statistics as JSON
        stats_json = {}
        for key, stats_dict in self.statistics.items():
            stats_json[key] = {
                str(k): asdict(v) for k, v in stats_dict.items()
            }
        
        json_path = self.output_dir / "trained_statistics.json"
        with open(json_path, 'w') as f:
            json.dump(stats_json, f, indent=2, default=str)
        console.print(f"[green]Saved statistics to {json_path}[/green]")
        
        # Save trade results as CSV
        trades_df = pd.DataFrame([asdict(t) for t in self.all_trades])
        csv_path = self.output_dir / "trade_results.csv"
        trades_df.to_csv(csv_path, index=False)
        console.print(f"[green]Saved trade results to {csv_path}[/green]")
        
        # Save parameters in scanner-compatible format
        scanner_params = self._create_scanner_parameters()
        params_path = self.output_dir / "scanner_parameters.json"
        with open(params_path, 'w') as f:
            json.dump(scanner_params, f, indent=2)
        console.print(f"[green]Saved scanner parameters to {params_path}[/green]")
        
        # Save as pickle for easy loading
        pickle_path = self.output_dir / "trained_parameters.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'statistics': self.statistics,
                'trades': self.all_trades,
                'signals': self.all_signals
            }, f)
        console.print(f"[green]Saved pickle file to {pickle_path}[/green]")
    
    def _create_scanner_parameters(self) -> Dict[str, Any]:
        """Create parameters in format compatible with scanner."""
        params = {
            'metadata': {
                'training_date': datetime.now().isoformat(),
                'lookback_months': self.lookback_months,
                'num_tickers': len(self.tickers),
                'num_trades': len(self.all_trades)
            },
            'priors': {},
            'setup_parameters': {}
        }
        
        # Overall prior
        overall_stats = list(self.statistics['overall'].values())[0]
        params['priors']['global'] = {
            'alpha': overall_stats.alpha,
            'beta': overall_stats.beta,
            'p_win': overall_stats.p_win,
            'avg_win': overall_stats.avg_win,
            'avg_loss': overall_stats.avg_loss
        }
        
        # Setup-specific parameters
        for (setup, ticker), stats in self.statistics['by_setup'].items():
            params['setup_parameters'][setup] = {
                'alpha': stats.alpha,
                'beta': stats.beta,
                'p_win': stats.p_win,
                'avg_win': stats.avg_win,
                'avg_loss': stats.avg_loss,
                'avg_r_win': stats.avg_r_win,
                'avg_r_loss': stats.avg_r_loss,
                'total_trades': stats.total_trades,
                'expectancy': stats.expectancy,
                'profit_factor': stats.profit_factor
            }
        
        return params
    
    def run(self) -> None:
        """Run the full training pipeline."""
        console.print(Panel.fit(
            "[bold cyan]Historical Parameter Trainer[/bold cyan]\n"
            f"Training on {len(self.tickers)} tickers\n"
            f"Lookback period: {self.lookback_months} months",
            border_style="cyan"
        ))
        
        # Step 1: Fetch data
        self.fetch_historical_data()
        
        # Step 2: Generate signals
        self.generate_all_signals()
        
        # Step 3: Simulate trades
        self.simulate_all_trades()
        
        # Step 4: Calculate statistics
        self.calculate_statistics()
        
        # Step 5: Display results
        self.display_results()
        
        # Step 6: Save parameters
        self.save_parameters()
        
        console.print("\n[bold green]✓ Training complete![/bold green]\n")


def main():
    """Main entry point."""
    # Example usage with common tickers
    tickers = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'INTC',
        # Finance
        'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK',
        # Consumer
        'WMT', 'HD', 'DIS', 'NKE', 'SBUX', 'MCD',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB',
        # Industrial
        'BA', 'CAT', 'GE', 'HON', 'MMM',
    ]
    
    # Initialize trainer
    trainer = HistoricalParameterTrainer(
        tickers=tickers,
        lookback_months=12,  # 12 months of history
        output_dir="trained_parameters"
    )
    
    # Run training
    trainer.run()


if __name__ == "__main__":
    main()



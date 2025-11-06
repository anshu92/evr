#!/usr/bin/env python3
"""
EVR Official Ticker Scanner with Full Framework

This scanner implements the complete EVR framework with:
1. Official ticker data from NASDAQ FTP
2. Comprehensive technical analysis
3. Empirical Bayes probability estimation
4. Kelly fraction sizing
5. Cost and slippage modeling
6. Risk management and circuit breakers
7. Expected growth ranking
8. Full TradePlan objects with all EVR fields
9. Optional ML classifier for probability estimation
10. Signal aggregation by default
"""

import json
import logging
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

import pandas as pd
import numpy as np
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler

# Optional ML imports for probability estimation
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, roc_auc_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class RateLimiter:
    """Advanced rate limiter with exponential backoff and jitter."""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.last_request_time = 0.0
        self.consecutive_failures = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if needed to respect rate limits."""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            # Only apply delay if we have recent failures
            # On success, allow parallel requests to proceed without delay
            if self.consecutive_failures > 0:
                delay = min(self.base_delay * (self.backoff_factor ** self.consecutive_failures), self.max_delay)
                # Add jitter to prevent thundering herd
                jitter = random.uniform(0.1, 0.3) * delay
                total_delay = delay + jitter
                
                if time_since_last < total_delay:
                    sleep_time = total_delay - time_since_last
                    time.sleep(sleep_time)
            # else: no delay on success - let threads proceed in parallel
            
            self.last_request_time = time.time()
    
    def record_success(self):
        """Record a successful request."""
        with self.lock:
            self.consecutive_failures = 0
    
    def record_failure(self):
        """Record a failed request."""
        with self.lock:
            self.consecutive_failures += 1


class ParallelDataFetcher:
    """Parallel data fetcher with rate limiting and retry logic."""
    
    def __init__(self, max_workers: int = 5, rate_limiter: Optional[RateLimiter] = None):
        self.max_workers = max_workers
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session = requests.Session()
        # Configure session for better performance
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_ticker_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch data for a single ticker with retry logic."""
        import yfinance as yf
        
        for attempt in range(3):
            try:
                self.rate_limiter.wait_if_needed()
                
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
                
                if not data.empty:
                    self.rate_limiter.record_success()
                    return data
                else:
                    self.rate_limiter.record_failure()
                    return None
                    
            except Exception as e:
                self.rate_limiter.record_failure()
                
                if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                    # Exponential backoff with jitter (capped at 8s to prevent long stalls)
                    wait_time = min(2 ** attempt * 2, 8) + random.uniform(0.5, 1.5)
                    time.sleep(wait_time)
                    continue
                else:
                    return None
        
        return None
    
    def fetch_multiple_tickers(self, tickers: List[str], period: str = "1y") -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch data for multiple tickers in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.fetch_ticker_data, ticker, period): ticker 
                for ticker in tickers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    results[ticker] = future.result()
                except Exception as e:
                    results[ticker] = None
        
        return results


@dataclass
class TradePlan:
    """EVR TradePlan object with all required fields."""
    ticker: str
    setup: str
    entry: float
    stop: float
    targets: List[float]
    p_win: float
    avg_r_win: float
    avg_r_loss: float
    expected_return: float
    kelly_fraction: float
    position_size: float  # Dollar value of position
    risk_dollars: float
    notes: str
    signal_type: str
    confidence: float
    take_profit: float
    cost_bps: float
    slippage_bps: float
    shares: int = 0  # Number of shares to buy
    # R-unit metrics (added dynamically)
    r_unit: float = 0.0
    reward_r: float = 0.0
    costs_r: float = 0.0
    expectancy_r: float = 0.0
    min_win_rate: float = 0.0
    required_win_rate: float = 0.0
    # Action: BUY/SHORT/NULL based on expectancy and win rate
    action: str = "NULL"
    # Liquidity score (0-1, higher is better)
    liquidity_score: float = 0.0


@dataclass
class Signal:
    """EVR Signal object."""
    symbol: str
    setup: str
    direction: int  # 1 for long, -1 for short
    strength: float
    timestamp: datetime
    features: Dict[str, float]


class RollingBayes:
    """Empirical Bayes probability estimation engine."""
    
    def __init__(self, window_size: int = 252, alpha: float = 1.0, beta: float = 1.0, 
                 ewma_decay: float = 0.1, use_regimes: bool = False, regime_window: int = 63,
                 portfolio_manager: Optional['PortfolioManager'] = None):
        """Initialize RollingBayes engine.
        
        Args:
            window_size: Rolling window size for historical data
            alpha: Beta prior alpha parameter
            beta: Beta prior beta parameter
            ewma_decay: EWMA decay factor for return tracking
            use_regimes: Whether to use volatility regime conditioning
            regime_window: Window size for regime calculation
            portfolio_manager: Portfolio manager for historical data
        """
        self.window_size = window_size
        self.alpha = alpha
        self.beta = beta
        self.ewma_decay = ewma_decay
        self.use_regimes = use_regimes
        self.regime_window = regime_window
        self.portfolio_manager = portfolio_manager
        
        # Storage for historical data
        self.counters = defaultdict(lambda: {
            'wins': 0, 'trades': 0, 'returns': [], 'timestamps': [],
            'avg_win': 0.0, 'avg_loss': 0.0, 'ewma_win': 0.0, 'ewma_loss': 0.0
        })
        
        # Regime storage
        self.regime_counters = defaultdict(lambda: defaultdict(lambda: {
            'wins': 0, 'trades': 0, 'returns': [], 'avg_win': 0.0, 'avg_loss': 0.0
        }))
        
        # Load historical data from portfolio if available
        if self.portfolio_manager:
            self._load_historical_data()
    
    def _load_historical_data(self) -> None:
        """Load historical trade data from portfolio manager."""
        if not self.portfolio_manager:
            return
        
        # Get closed positions from portfolio
        closed_positions = [p for p in self.portfolio_manager.state.positions if p.status != "OPEN"]
        
        for position in closed_positions:
            if position.pnl is not None and position.exit_date is not None:
                # Calculate return percentage
                return_pct = position.return_pct if position.return_pct is not None else 0.0
                
                # Determine if it was a win
                is_win = return_pct > 0
                
                # Update counters
                key = self._get_key(position.setup, position.ticker)
                counter = self.counters[key]
                
                counter['trades'] += 1
                if is_win:
                    counter['wins'] += 1
                
                counter['returns'].append(return_pct)
                counter['timestamps'].append(position.exit_date)
                
                # Update averages
                if is_win:
                    counter['avg_win'] = sum([r for r in counter['returns'] if r > 0]) / len([r for r in counter['returns'] if r > 0])
                else:
                    counter['avg_loss'] = sum([r for r in counter['returns'] if r < 0]) / len([r for r in counter['returns'] if r < 0])
    
    def _get_key(self, setup: str, ticker: str, timeframe: str = '1d') -> Tuple[str, str, str]:
        """Get storage key for counters."""
        return (setup, ticker, timeframe)
    
    def _get_regime(self, volatility: float) -> str:
        """Determine volatility regime."""
        if volatility < 0.15:  # Low volatility
            return 'low'
        elif volatility < 0.30:  # Medium volatility
            return 'medium'
        else:  # High volatility
            return 'high'
    
    def update(self, setup: str, ticker: str, timeframe: str, result: float, 
               volatility: float = None, timestamp: datetime = None) -> None:
        """Update counters with trade result.
        
        Args:
            setup: Trading setup name
            ticker: Stock ticker
            timeframe: Timeframe (e.g., '1d')
            result: Trade return (positive for win, negative for loss)
            volatility: Current volatility (for regime conditioning)
            timestamp: Trade timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        key = self._get_key(setup, ticker, timeframe)
        counter = self.counters[key]
        
        # Update basic counters
        counter['trades'] += 1
        counter['returns'].append(result)
        counter['timestamps'].append(timestamp)
        
        if result > 0:
            counter['wins'] += 1
        
        # Maintain rolling window
        if len(counter['returns']) > self.window_size:
            old_return = counter['returns'].pop(0)
            old_timestamp = counter['timestamps'].pop(0)
            if old_return > 0:
                counter['wins'] -= 1
        
        # Update EWMA for win/loss returns
        if result > 0:
            counter['ewma_win'] = (1 - self.ewma_decay) * counter['ewma_win'] + self.ewma_decay * result
        else:
            counter['ewma_loss'] = (1 - self.ewma_decay) * counter['ewma_loss'] + self.ewma_decay * result
        
        # Update regime counters if enabled
        if self.use_regimes and volatility is not None:
            regime = self._get_regime(volatility)
            regime_key = self._get_key(setup, ticker, timeframe)
            regime_counter = self.regime_counters[regime][regime_key]
            
            regime_counter['trades'] += 1
            regime_counter['returns'].append(result)
            
            if result > 0:
                regime_counter['wins'] += 1
            
            # Maintain regime window
            if len(regime_counter['returns']) > self.regime_window:
                old_return = regime_counter['returns'].pop(0)
                if old_return > 0:
                    regime_counter['wins'] -= 1
    
    def estimate(self, setup: str, ticker: str, timeframe: str = '1d', 
                 volatility: float = None) -> Tuple[float, float, float]:
        """Estimate probability and return parameters.
        
        Args:
            setup: Trading setup name
            ticker: Stock ticker
            timeframe: Timeframe
            volatility: Current volatility (for regime conditioning)
            
        Returns:
            Tuple of (p_hat, avg_win, avg_loss)
        """
        key = self._get_key(setup, ticker, timeframe)
        
        # Use regime-specific estimates if enabled
        if self.use_regimes and volatility is not None:
            regime = self._get_regime(volatility)
            regime_counter = self.regime_counters[regime][key]
            
            if regime_counter['trades'] >= 10:  # Minimum sample size
                wins = regime_counter['wins']
                trades = regime_counter['trades']
                returns = regime_counter['returns']
            else:
                # Fall back to overall estimates
                counter = self.counters[key]
                wins = counter['wins']
                trades = counter['trades']
                returns = counter['returns']
        else:
            counter = self.counters[key]
            wins = counter['wins']
            trades = counter['trades']
            returns = counter['returns']
        
        # Beta-Binomial smoothing
        if trades == 0:
            # No historical data - use default estimates
            p_hat = 0.5  # 50% win rate assumption
            avg_win = 0.05  # 5% average win
            avg_loss = -0.03  # 3% average loss
        else:
            p_hat = (wins + self.alpha) / (trades + self.alpha + self.beta)
            
            # Calculate average win/loss
            win_returns = [r for r in returns if r > 0]
            loss_returns = [r for r in returns if r < 0]
            
            avg_win = sum(win_returns) / len(win_returns) if win_returns else 0.05
            avg_loss = sum(loss_returns) / len(loss_returns) if loss_returns else -0.03
        
        return p_hat, avg_win, avg_loss
    
    def get_ewma_estimates(self, setup: str, ticker: str, timeframe: str = '1d') -> Tuple[float, float]:
        """Get EWMA-based return estimates.
        
        Returns:
            Tuple of (ewma_win, ewma_loss)
        """
        key = self._get_key(setup, ticker, timeframe)
        counter = self.counters[key]
        return counter['ewma_win'], counter['ewma_loss']


class FeatureClassifier:
    """Feature-based classifier for probability estimation."""
    
    def __init__(self, use_calibration: bool = True, retrain_frequency: int = 252):
        """Initialize feature classifier.
        
        Args:
            use_calibration: Whether to use isotonic calibration
            retrain_frequency: How often to retrain (in periods)
        """
        self.use_calibration = use_calibration
        self.retrain_frequency = retrain_frequency
        
        if not ML_AVAILABLE:
            raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")
        
        # Initialize models
        self.base_model = LogisticRegression(random_state=42, max_iter=1000)
        
        if use_calibration:
            self.model = CalibratedClassifierCV(self.base_model, method='isotonic', cv=3)
        else:
            self.model = self.base_model
        
        # Training data storage
        self.features_history = []
        self.outcomes_history = []
        self.timestamps_history = []
        
        # Performance tracking
        self.brier_scores = []
        self.roc_auc_scores = []
        
        self.last_retrain = datetime.now()
    
    def add_training_data(self, features: Dict[str, float], outcome: bool, timestamp: datetime = None) -> None:
        """Add training data point.
        
        Args:
            features: Feature dictionary
            outcome: Whether trade was successful
            timestamp: Data timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.features_history.append(features)
        self.outcomes_history.append(outcome)
        self.timestamps_history.append(timestamp)
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for model input."""
        # Convert to consistent feature vector
        feature_names = [
            'rsi', 'macd', 'bb_position', 'volume_ratio', 'momentum_5', 'momentum_10',
            'volatility_20', 'stoch_k', 'williams_r', 'cci', 'atr_percentage', 'adx'
        ]
        
        feature_vector = []
        for name in feature_names:
            value = features.get(name, 0.0)
            # Handle NaN/inf values
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            feature_vector.append(value)
        
        return np.array(feature_vector).reshape(1, -1)
    
    def train(self) -> None:
        """Train the classifier."""
        if len(self.features_history) < 50:  # Minimum training samples
            return
        
        # Prepare training data
        X = []
        y = []
        
        for features, outcome in zip(self.features_history, self.outcomes_history):
            feature_vector = self._prepare_features(features)
            X.append(feature_vector.flatten())
            y.append(outcome)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        self.model.fit(X, y)
        
        # Evaluate performance
        if len(y) > 10:
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            
            # Brier score
            brier_score = brier_score_loss(y, y_pred_proba)
            self.brier_scores.append(brier_score)
            
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(y, y_pred_proba)
                self.roc_auc_scores.append(roc_auc)
            except ValueError:
                pass  # Handle case where only one class present
        
        self.last_retrain = datetime.now()
    
    def predict_probability(self, features: Dict[str, float]) -> float:
        """Predict probability of success.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Probability of success (0-1)
        """
        if len(self.features_history) < 10:
            return 0.5  # Default probability
        
        # Check if retraining is needed
        if (datetime.now() - self.last_retrain).days >= self.retrain_frequency:
            self.train()
        
        # Prepare features and predict
        feature_vector = self._prepare_features(features)
        
        try:
            prob = self.model.predict_proba(feature_vector)[0, 1]
            return float(prob)
        except:
            return 0.5  # Fallback to default


class CostModel:
    """Cost and slippage modeling."""
    
    def __init__(self, commission_bps: float = 1.0, slippage_bps: float = 5.0, 
                 slippage_atr_multiplier: float = 0.5):
        """Initialize cost model.
        
        Args:
            commission_bps: Commission per trade in basis points
            slippage_bps: Fixed slippage in basis points
            slippage_atr_multiplier: ATR-based slippage multiplier
        """
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.slippage_atr_multiplier = slippage_atr_multiplier
    
    def calculate_costs(self, entry_price: float, atr_percentage: float, 
                      position_size: float) -> Tuple[float, float]:
        """Calculate total costs for a trade.
        
        Args:
            entry_price: Entry price
            atr_percentage: ATR as percentage of price
            position_size: Position size in dollars
            
        Returns:
            Tuple of (commission_cost, slippage_cost)
        """
        # Commission cost (based on position size, not entry price)
        commission_cost = position_size * (self.commission_bps / 10000)
        
        # Slippage cost (fixed + ATR-based, based on position size)
        fixed_slippage = position_size * (self.slippage_bps / 10000)
        atr_slippage = position_size * (atr_percentage / 100) * self.slippage_atr_multiplier
        slippage_cost = fixed_slippage + atr_slippage
        
        return commission_cost, slippage_cost


class KellySizing:
    """Kelly fraction sizing calculation."""
    
    def __init__(self, kelly_fraction: float = 0.25, max_kelly_fraction: float = 0.5,
                 max_position_size: float = 0.1, min_position_size: float = 0.001):
        """Initialize Kelly sizing.
        
        Args:
            kelly_fraction: Fraction of Kelly to use (e.g., 0.25 for quarter-Kelly)
            max_kelly_fraction: Maximum Kelly fraction allowed
            max_position_size: Maximum position size as fraction of equity
            min_position_size: Minimum position size as fraction of equity
        """
        self.kelly_fraction = kelly_fraction
        self.max_kelly_fraction = max_kelly_fraction
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
    
    def calculate_kelly_fraction(self, p_win: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly fraction.
        
        Args:
            p_win: Probability of winning
            avg_win: Average win return
            avg_loss: Average loss return (should be negative)
            
        Returns:
            Kelly fraction (0-1)
        """
        if avg_loss == 0 or p_win <= 0 or p_win >= 1:
            return 0.0
        
        # Kelly formula: f* = (b*p - q) / b, where b = avg_win/|avg_loss|
        b = avg_win / abs(avg_loss)
        q = 1 - p_win
        
        kelly = (b * p_win - q) / b
        
        # Apply constraints
        kelly = max(0, min(kelly, self.max_kelly_fraction))
        
        # Apply fractional Kelly
        kelly *= self.kelly_fraction
        
        return kelly
    
    def size_position(self, equity: float, entry_price: float, stop_price: float,
                      kelly_fraction: float) -> Tuple[float, float]:
        """Calculate position size.
        
        Args:
            equity: Available equity
            entry_price: Entry price
            stop_price: Stop loss price
            kelly_fraction: Kelly fraction
            
        Returns:
            Tuple of (position_size_dollars, shares)
        """
        # Risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share == 0:
            return 0.0, 0
        
        # Position size based on Kelly
        position_size_dollars = kelly_fraction * equity
        
        # Apply position size constraints
        max_position_dollars = self.max_position_size * equity
        min_position_dollars = self.min_position_size * equity
        
        position_size_dollars = max(min_position_dollars, 
                                  min(position_size_dollars, max_position_dollars))
        
        # Calculate shares
        shares = int(position_size_dollars / entry_price)
        
        # Adjust for actual position size
        actual_position_size = shares * entry_price
        
        return actual_position_size, shares


class RiskGuards:
    """Risk management and circuit breakers."""
    
    def __init__(self, initial_capital: float = 100000, max_position_size: float = 0.1,
                 daily_loss_limit: float = 0.03, weekly_loss_limit: float = 0.08,
                 max_drawdown_limit: float = 0.15, max_positions: int = 20):
        """Initialize risk guards.
        
        Args:
            initial_capital: Initial capital
            max_position_size: Maximum position size as fraction of equity
            daily_loss_limit: Daily loss limit as fraction of equity
            weekly_loss_limit: Weekly loss limit as fraction of equity
            max_drawdown_limit: Maximum drawdown limit as fraction of equity
            max_positions: Maximum number of concurrent positions
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.max_positions = max_positions
        
        # Risk tracking
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.peak_capital = initial_capital
        self.last_reset_date = datetime.now().date()
        
        # Circuit breakers
        self.daily_loss_breached = False
        self.weekly_loss_breached = False
        self.drawdown_breached = False
    
    def check_trade_plan(self, trade_plan: TradePlan) -> Dict[str, Any]:
        """Check if trade plan passes risk guards.
        
        Args:
            trade_plan: Trade plan to check
            
        Returns:
            Dictionary with check results
        """
        checks = {
            'position_size_ok': True,
            'daily_loss_ok': True,
            'weekly_loss_ok': True,
            'drawdown_ok': True,
            'max_positions_ok': True,
            'all_checks_passed': True
        }
        
        # Check position size
        position_fraction = trade_plan.position_size / self.current_capital
        if position_fraction > self.max_position_size:
            checks['position_size_ok'] = False
            checks['all_checks_passed'] = False
        
        # Check daily loss limit
        if self.daily_loss_breached:
            checks['daily_loss_ok'] = False
            checks['all_checks_passed'] = False
        
        # Check weekly loss limit
        if self.weekly_loss_breached:
            checks['weekly_loss_ok'] = False
            checks['all_checks_passed'] = False
        
        # Check drawdown limit
        if self.drawdown_breached:
            checks['drawdown_ok'] = False
            checks['all_checks_passed'] = False
        
        # Check max positions
        if len(self.current_positions) >= self.max_positions:
            checks['max_positions_ok'] = False
            checks['all_checks_passed'] = False
        
        return checks
    
    def update_pnl(self, pnl: float) -> None:
        """Update P&L and check circuit breakers."""
        self.current_capital += pnl
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Check circuit breakers
        self._check_circuit_breakers()
    
    def _check_circuit_breakers(self) -> None:
        """Check and update circuit breaker status."""
        # Daily loss breaker
        if self.daily_pnl < -self.daily_loss_limit * self.current_capital:
            self.daily_loss_breached = True
        
        # Weekly loss breaker
        if self.weekly_pnl < -self.weekly_loss_limit * self.current_capital:
            self.weekly_loss_breached = True
        
        # Drawdown breaker
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.max_drawdown_limit:
            self.drawdown_breached = True
    
    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.daily_loss_breached = False
    
    def reset_weekly(self) -> None:
        """Reset weekly counters."""
        self.weekly_pnl = 0.0
        self.weekly_loss_breached = False


class LiquidityGuards:
    """Liquidity guardrails to ensure tradeable stocks."""
    
    def __init__(self, min_avg_volume: int = 100000, min_price: float = 1.0, 
                 max_price: float = 10000.0, min_market_cap: float = 100000000,
                 max_bid_ask_spread: float = 0.05, min_daily_volume: int = 50000):
        """Initialize liquidity guards.
        
        Args:
            min_avg_volume: Minimum average daily volume (30-day)
            min_price: Minimum stock price to trade
            max_price: Maximum stock price to trade
            min_market_cap: Minimum market capitalization
            max_bid_ask_spread: Maximum bid-ask spread as percentage of price
            min_daily_volume: Minimum daily volume for current day
        """
        self.min_avg_volume = min_avg_volume
        self.min_price = min_price
        self.max_price = max_price
        self.min_market_cap = min_market_cap
        self.max_bid_ask_spread = max_bid_ask_spread
        self.min_daily_volume = min_daily_volume
    
    def check_liquidity(self, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Check if stock meets liquidity requirements.
        
        Args:
            data: Stock data DataFrame
            current_price: Current stock price
            
        Returns:
            Dictionary with liquidity checks and overall result
        """
        checks = {
            'price_ok': True,
            'volume_ok': True,
            'avg_volume_ok': True,
            'daily_volume_ok': True,
            'spread_ok': True,
            'all_checks_passed': True
        }
        
        try:
            # Check price range
            if current_price < self.min_price or current_price > self.max_price:
                checks['price_ok'] = False
                checks['all_checks_passed'] = False
            
            # Check daily volume (if available)
            if len(data) > 0:
                latest_volume = data['Volume'].iloc[-1]
                if latest_volume < self.min_daily_volume:
                    checks['daily_volume_ok'] = False
                    checks['all_checks_passed'] = False
                
                # Check average volume (30-day)
                if len(data) >= 30:
                    avg_volume = data['Volume'].tail(30).mean()
                    if avg_volume < self.min_avg_volume:
                        checks['avg_volume_ok'] = False
                        checks['all_checks_passed'] = False
                elif len(data) >= 10:
                    # Use available data if less than 30 days
                    avg_volume = data['Volume'].mean()
                    if avg_volume < self.min_avg_volume * 0.5:  # Relaxed requirement
                        checks['avg_volume_ok'] = False
                        checks['all_checks_passed'] = False
                
                # Estimate bid-ask spread (simplified using high-low range)
                if 'High' in data.columns and 'Low' in data.columns:
                    latest_high = data['High'].iloc[-1]
                    latest_low = data['Low'].iloc[-1]
                    if latest_high > 0 and latest_low > 0:
                        estimated_spread = (latest_high - latest_low) / current_price
                        if estimated_spread > self.max_bid_ask_spread:
                            checks['spread_ok'] = False
                            checks['all_checks_passed'] = False
            
        except Exception as e:
            # If we can't calculate liquidity metrics, fail the check
            checks['all_checks_passed'] = False
        
        return checks
    
    def get_liquidity_score(self, data: pd.DataFrame, current_price: float) -> float:
        """Calculate liquidity score (0-1, higher is better).
        
        Args:
            data: Stock data DataFrame
            current_price: Current stock price
            
        Returns:
            Liquidity score between 0 and 1
        """
        try:
            score = 1.0
            
            # Volume score (0-0.4)
            if len(data) > 0:
                latest_volume = data['Volume'].iloc[-1]
                volume_score = min(1.0, latest_volume / (self.min_avg_volume * 2))
                score *= (0.2 + 0.2 * volume_score)
                
                # Average volume score (0-0.3)
                if len(data) >= 10:
                    avg_volume = data['Volume'].tail(min(30, len(data))).mean()
                    avg_volume_score = min(1.0, avg_volume / self.min_avg_volume)
                    score *= (0.1 + 0.2 * avg_volume_score)
            
            # Price score (0-0.2)
            if self.min_price <= current_price <= self.max_price:
                price_score = 0.2
            else:
                price_score = 0.0
            score *= price_score
            
            # Spread score (0-0.1)
            if 'High' in data.columns and 'Low' in data.columns and len(data) > 0:
                latest_high = data['High'].iloc[-1]
                latest_low = data['Low'].iloc[-1]
                if latest_high > 0 and latest_low > 0:
                    estimated_spread = (latest_high - latest_low) / current_price
                    spread_score = max(0.0, 1.0 - estimated_spread / self.max_bid_ask_spread)
                    score *= (0.05 + 0.05 * spread_score)
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.0


@dataclass
class PortfolioPosition:
    """Portfolio position tracking."""
    ticker: str
    setup: str
    entry_price: float
    stop_price: float
    target_prices: List[float]
    position_size: float  # Number of shares
    shares: int = 0  # Explicit number of shares field
    entry_date: datetime = None
    p_win: float = 0.0
    expected_return: float = 0.0
    kelly_fraction: float = 0.0
    risk_dollars: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED_OUT, TARGET_HIT
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    pnl: Optional[float] = None
    return_pct: Optional[float] = None


@dataclass
class PortfolioState:
    """Portfolio state tracking."""
    total_capital: float
    available_capital: float
    allocated_capital: float
    total_pnl: float
    total_return_pct: float
    positions: List[PortfolioPosition]
    last_updated: datetime
    run_count: int
    performance_history: List[Dict[str, Any]]


class PortfolioManager:
    """Portfolio management system with $1000 allocation and performance tracking."""
    
    def __init__(self, initial_capital: float = 1000.0, data_file: str = "portfolio_state.json"):
        """Initialize portfolio manager.
        
        Args:
            initial_capital: Initial capital allocation
            data_file: File to persist portfolio state
        """
        self.initial_capital = initial_capital
        self.data_file = Path(data_file)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state = self._load_state()
        
    def _load_state(self) -> PortfolioState:
        """Load portfolio state from file or create new."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Convert positions back to PortfolioPosition objects
                positions = []
                for pos_data in data.get('positions', []):
                    # Handle backward compatibility: if 'shares' field doesn't exist, set it from position_size
                    if 'shares' not in pos_data:
                        pos_data['shares'] = int(pos_data.get('position_size', 0))
                    pos = PortfolioPosition(**pos_data)
                    # Convert string dates back to datetime
                    if isinstance(pos.entry_date, str):
                        pos.entry_date = datetime.fromisoformat(pos.entry_date)
                    if isinstance(pos.exit_date, str):
                        pos.exit_date = datetime.fromisoformat(pos.exit_date)
                    positions.append(pos)
                
                # Convert last_updated string to datetime
                last_updated = data.get('last_updated')
                if isinstance(last_updated, str):
                    last_updated = datetime.fromisoformat(last_updated)
                
                return PortfolioState(
                    total_capital=data.get('total_capital', self.initial_capital),
                    available_capital=data.get('available_capital', self.initial_capital),
                    allocated_capital=data.get('allocated_capital', 0.0),
                    total_pnl=data.get('total_pnl', 0.0),
                    total_return_pct=data.get('total_return_pct', 0.0),
                    positions=positions,
                    last_updated=last_updated or datetime.now(),
                    run_count=data.get('run_count', 0),
                    performance_history=data.get('performance_history', [])
                )
            except Exception as e:
                print(f"Error loading portfolio state: {e}")
                return self._create_initial_state()
        else:
            return self._create_initial_state()
    
    def _create_initial_state(self) -> PortfolioState:
        """Create initial portfolio state."""
        return PortfolioState(
            total_capital=self.initial_capital,
            available_capital=self.initial_capital,
            allocated_capital=0.0,
            total_pnl=0.0,
            total_return_pct=0.0,
            positions=[],
            last_updated=datetime.now(),
            run_count=0,
            performance_history=[]
        )
    
    def _save_state(self) -> None:
        """Save portfolio state to file (only open positions)."""
        try:
            # Convert to serializable format
            state_dict = asdict(self.state)
            
            # Filter to only keep open positions
            open_positions = [pos for pos in state_dict['positions'] if pos['status'] == 'OPEN']
            state_dict['positions'] = open_positions
            
            # Convert datetime objects to strings
            state_dict['last_updated'] = self.state.last_updated.isoformat()
            for pos in state_dict['positions']:
                if pos['entry_date']:
                    pos['entry_date'] = pos['entry_date'].isoformat()
                if pos['exit_date']:
                    pos['exit_date'] = pos['exit_date'].isoformat()
            
            with open(self.data_file, 'w') as f:
                json.dump(state_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving portfolio state: {e}")
    
    def update_allocation(self, new_capital: float) -> None:
        """Update the total capital allocation."""
        self.state.total_capital = new_capital
        self.state.available_capital = new_capital - self.state.allocated_capital
        self.state.last_updated = datetime.now()
        self._save_state()
    
    def increment_run_count(self) -> None:
        """Increment run count and update timestamp."""
        self.state.run_count += 1
        self.state.last_updated = datetime.now()
        self._save_state()
    
    def add_position(self, trade_plan: TradePlan) -> bool:
        """Add a new position to the portfolio.
        
        Returns:
            bool: True if position added successfully, False otherwise
        """
        # Check if position already exists for this ticker/setup
        existing_position = self._find_existing_position(trade_plan.ticker, trade_plan.setup)
        if existing_position:
            self.logger.debug(f"Position already exists for {trade_plan.ticker} ({trade_plan.setup}), skipping duplicate")
            return False
        
        # Check if we have enough capital (position_size is in dollars)
        if self.state.available_capital < trade_plan.position_size:
            return False
        
        position = PortfolioPosition(
            ticker=trade_plan.ticker,
            setup=trade_plan.setup,
            entry_price=trade_plan.entry,
            stop_price=trade_plan.stop,
            target_prices=trade_plan.targets,
            position_size=trade_plan.position_size,  # Dollar value
            shares=trade_plan.shares,  # Number of shares
            entry_date=datetime.now(),
            p_win=trade_plan.p_win,
            expected_return=trade_plan.expected_return,
            kelly_fraction=trade_plan.kelly_fraction,
            risk_dollars=trade_plan.risk_dollars
        )
        
        self.state.positions.append(position)
        # Deduct the dollar value of the position from available capital
        self.state.allocated_capital += trade_plan.position_size
        self.state.available_capital -= trade_plan.position_size
        self._save_state()
        return True
    
    def _find_existing_position(self, ticker: str, setup: str) -> Optional[PortfolioPosition]:
        """Find existing open position for ticker/setup combination.
        
        Args:
            ticker: Stock ticker
            setup: Trading setup name
            
        Returns:
            PortfolioPosition if found, None otherwise
        """
        for position in self.state.positions:
            if position.status == "OPEN" and position.ticker == ticker and position.setup == setup:
                return position
        return None
    
    def _calculate_atr_percentage(self, data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """ATR percentage from price data."""
        if data is None or data.empty:
            return None
        
        required = {'High', 'Low', 'Close'}
        if not required.issubset(data.columns):
            return None
        
        if len(data) <= period:
            return None
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        current_price = float(close.iloc[-1])
        if current_price <= 0:
            return None
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = true_range.rolling(period).mean()
        atr_value = atr_series.iloc[-1]
        
        if pd.isna(atr_value) or atr_value <= 0:
            return None
        
        return float((atr_value / current_price) * 100)
    
    def _update_position_levels_from_data(self, position: PortfolioPosition, data: pd.DataFrame) -> bool:
        """Refresh stop and targets for a position."""
        atr_pct = self._calculate_atr_percentage(data)
        if atr_pct is None:
            return False
        
        close_series = data['Close']
        current_price = float(close_series.iloc[-1])
        if current_price <= 0:
            return False
        
        is_long = position.entry_price > position.stop_price
        stop_multiplier = 2.0
        target_multiplier = 4.0
        
        if is_long:
            candidate_stop = current_price * (1 - (atr_pct / 100.0) * stop_multiplier)
            candidate_target = current_price * (1 + (atr_pct / 100.0) * target_multiplier)
            new_stop = max(position.stop_price, candidate_stop)
            existing_target = position.target_prices[0] if position.target_prices else candidate_target
            new_target = max(existing_target, candidate_target)
        else:
            candidate_stop = current_price * (1 + (atr_pct / 100.0) * stop_multiplier)
            candidate_target = current_price * (1 - (atr_pct / 100.0) * target_multiplier)
            new_stop = min(position.stop_price, candidate_stop)
            existing_target = position.target_prices[0] if position.target_prices else candidate_target
            new_target = min(existing_target, candidate_target)
        
        updated = False
        
        if abs(new_stop - position.stop_price) > 1e-4:
            old_stop = position.stop_price
            position.stop_price = float(new_stop)
            updated = True
            self.logger.info(
                f"Updated stop for {position.ticker}: {old_stop:.4f} -> {position.stop_price:.4f}"
            )
        
        targets = list(position.target_prices) if position.target_prices else []
        if not targets:
            targets = [float(new_target)]
            updated = True
        else:
            if abs(targets[0] - new_target) > 1e-4:
                old_target = targets[0]
                targets[0] = float(new_target)
                updated = True
                self.logger.info(
                    f"Updated target for {position.ticker}: {old_target:.4f} -> {targets[0]:.4f}"
                )
        position.target_prices = targets
        
        if updated:
            if position.shares > 0:
                if is_long:
                    risk_per_share = max(position.entry_price - position.stop_price, 0.0)
                else:
                    risk_per_share = max(position.stop_price - position.entry_price, 0.0)
                position.risk_dollars = risk_per_share * position.shares
            else:
                position.risk_dollars = 0.0
            
            if position.entry_price > 0:
                if is_long:
                    potential_return = max(targets[0] - position.entry_price, 0.0) / position.entry_price
                    potential_loss = max(position.entry_price - position.stop_price, 0.0) / position.entry_price
                else:
                    potential_return = max(position.entry_price - targets[0], 0.0) / position.entry_price
                    potential_loss = max(position.stop_price - position.entry_price, 0.0) / position.entry_price
                position.expected_return = position.p_win * potential_return - (1 - position.p_win) * potential_loss
            else:
                position.expected_return = 0.0
        
        return updated
    
    def monitor_and_close_positions(self, data_fetcher, max_holding_days: int = 7) -> Dict[str, Any]:
        """Monitor all open positions and automatically close those that hit stops, targets, or exceed holding period.
        
        Args:
            data_fetcher: ParallelDataFetcher instance to fetch current prices
            max_holding_days: Maximum days to hold a position (default: 7 days)
            
        Returns:
            Dictionary with monitoring results
        """
        open_positions = [p for p in self.state.positions if p.status == "OPEN"]
        
        if not open_positions:
            return {
                'monitored': 0,
                'closed': 0,
                'stopped_out': 0,
                'targets_hit': 0,
                'updated': 0,
                'time_exited': 0,
                'errors': 0
            }
        
        # Fetch current data for all open positions
        tickers = [p.ticker for p in open_positions]
        self.logger.info(f"Monitoring {len(tickers)} open positions: {', '.join(tickers)}")
        
        ticker_data = data_fetcher.fetch_multiple_tickers(tickers, period="1mo")
        
        results = {
            'monitored': len(tickers),
            'closed': 0,
            'stopped_out': 0,
            'targets_hit': 0,
            'time_exited': 0,
            'updated': 0,
            'errors': 0
        }
        
        updates_made = False
        
        for position in open_positions:
            data = ticker_data.get(position.ticker)
            
            if data is None or data.empty:
                self.logger.warning(f"Could not fetch data for {position.ticker}, skipping position check")
                results['errors'] += 1
                continue
            
            try:
                # Get current price (latest close)
                current_price = float(data['Close'].iloc[-1])
                
                # Check if position has exceeded max holding period (Option 3: Time-based exit)
                days_held = (datetime.now() - position.entry_date).days
                if days_held >= max_holding_days:
                    closed_pos = self.close_position(position.ticker, current_price, "TIME_EXIT")
                    if closed_pos:
                        results['closed'] += 1
                        results['time_exited'] += 1
                        pnl_str = f"${closed_pos.pnl:+,.2f}" if closed_pos.pnl else "N/A"
                        self.logger.info(
                            f"⏰ Time exit {position.ticker} @ ${current_price:.2f} (held {days_held} days) - "
                            f"Entry: ${position.entry_price:.2f}, P&L: {pnl_str}"
                        )
                    continue  # Skip stop/target checks if time-exited
                
                if self._update_position_levels_from_data(position, data):
                    results['updated'] += 1
                    updates_made = True
                
                # Determine if long or short based on entry/stop relationship
                is_long = position.entry_price > position.stop_price
                
                should_close = False
                reason = None
                
                if is_long:
                    # Long position checks
                    if current_price <= position.stop_price:
                        should_close = True
                        reason = "STOPPED_OUT"
                        results['stopped_out'] += 1
                    elif position.target_prices and current_price >= position.target_prices[0]:
                        should_close = True
                        reason = "TARGET_HIT"
                        results['targets_hit'] += 1
                else:
                    # Short position checks
                    if current_price >= position.stop_price:
                        should_close = True
                        reason = "STOPPED_OUT"
                        results['stopped_out'] += 1
                    elif position.target_prices and current_price <= position.target_prices[0]:
                        should_close = True
                        reason = "TARGET_HIT"
                        results['targets_hit'] += 1
                
                if should_close:
                    closed_pos = self.close_position(position.ticker, current_price, reason)
                    if closed_pos:
                        results['closed'] += 1
                        pnl_str = f"${closed_pos.pnl:+,.2f}" if closed_pos.pnl else "N/A"
                        self.logger.info(
                            f"✓ Closed {position.ticker} @ ${current_price:.2f} ({reason}) - "
                            f"Entry: ${position.entry_price:.2f}, P&L: {pnl_str}"
                        )
                
            except Exception as e:
                self.logger.error(f"Error monitoring position {position.ticker}: {e}")
                results['errors'] += 1
        
        if updates_made:
            self.state.last_updated = datetime.now()
            self._save_state()
        
        return results
    
    def close_position(self, ticker: str, exit_price: float, reason: str = "MANUAL") -> Optional[PortfolioPosition]:
        """Close a position and calculate P&L."""
        for position in self.state.positions:
            if position.ticker == ticker and position.status == "OPEN":
                position.exit_price = exit_price
                position.exit_date = datetime.now()
                position.status = reason
                
                # Calculate P&L based on shares
                position.pnl = (exit_price - position.entry_price) * position.shares
                position.return_pct = (exit_price - position.entry_price) / position.entry_price
                
                # Update portfolio totals
                self.state.total_pnl += position.pnl
                self.state.total_capital += position.pnl
                # Return the original position_size (dollars) plus P&L to available capital
                self.state.available_capital += position.position_size + position.pnl
                self.state.allocated_capital -= position.position_size
                
                # Update return percentage
                self.state.total_return_pct = (self.state.total_capital - self.initial_capital) / self.initial_capital
                
                # Record performance
                self._record_performance()
                self._save_state()
                return position
        
        return None
    
    def _record_performance(self) -> None:
        """Record current performance metrics."""
        performance = {
            'timestamp': datetime.now().isoformat(),
            'total_capital': self.state.total_capital,
            'total_pnl': self.state.total_pnl,
            'total_return_pct': self.state.total_return_pct,
            'open_positions': len([p for p in self.state.positions if p.status == "OPEN"]),
            'run_count': self.state.run_count
        }
        self.state.performance_history.append(performance)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary statistics."""
        open_positions = [p for p in self.state.positions if p.status == "OPEN"]
        closed_positions = [p for p in self.state.positions if p.status != "OPEN"]
        
        # Calculate win rate
        wins = len([p for p in closed_positions if p.pnl and p.pnl > 0])
        total_trades = len(closed_positions)
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        
        # Calculate average win/loss
        win_pnls = [p.pnl for p in closed_positions if p.pnl and p.pnl > 0]
        loss_pnls = [p.pnl for p in closed_positions if p.pnl and p.pnl < 0]
        
        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
        
        return {
            'total_capital': self.state.total_capital,
            'available_capital': self.state.available_capital,
            'allocated_capital': self.state.allocated_capital,
            'total_pnl': self.state.total_pnl,
            'total_return_pct': self.state.total_return_pct,
            'open_positions': len(open_positions),
            'closed_positions': len(closed_positions),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'run_count': self.state.run_count,
            'last_updated': self.state.last_updated
        }
    
    def get_historical_performance(self) -> List[Dict[str, Any]]:
        """Get historical performance data."""
        return self.state.performance_history


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
    """Official ticker scanner with full EVR framework integration."""
    
    def __init__(self, log_level: str = "INFO", use_ml_classifier: bool = True,
                 initial_capital: float = 1000, min_avg_volume: int = 100000,
                 min_price: float = 1.0, max_price: float = 10000.0,
                 max_bid_ask_spread: float = 0.05, min_daily_volume: int = 50000,
                 portfolio_file: str = "portfolio_state.json"):
        """Initialize the scanner with EVR framework.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            use_ml_classifier: Whether to use ML classifier for probability estimation (Bayesian calibration)
            initial_capital: Initial capital for risk management
            min_avg_volume: Minimum average daily volume for liquidity
            min_price: Minimum stock price for liquidity
            max_price: Maximum stock price for liquidity
            max_bid_ask_spread: Maximum bid-ask spread as percentage
            min_daily_volume: Minimum daily volume for current day
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
        
        # Delisted tickers tracking
        self.delisted_file = self.cache_dir / "delisted_tickers.json"
        self.delisted_tickers = self._load_delisted_tickers()
        
        # Initialize rate limiter and parallel data fetcher
        self.rate_limiter = RateLimiter(base_delay=0.5, max_delay=10.0, backoff_factor=1.5)
        self.data_fetcher = ParallelDataFetcher(max_workers=8, rate_limiter=self.rate_limiter)
        
        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(
            initial_capital=initial_capital,
            data_file=portfolio_file
        )
        
        # Initialize EVR framework components with portfolio manager
        self.prob_model = RollingBayes(
            window_size=252,
            alpha=1.0,
            beta=1.0,
            ewma_decay=0.1,
            use_regimes=True,
            regime_window=63,
            portfolio_manager=self.portfolio_manager
        )
        
        # Load trained parameters if available
        self._load_trained_parameters()
        
        # Initialize ML classifier if requested and available
        self.use_ml_classifier = use_ml_classifier and ML_AVAILABLE
        if self.use_ml_classifier:
            self.ml_classifier = FeatureClassifier(
                use_calibration=True,
                retrain_frequency=252
            )
            self.logger.info("ML classifier enabled")
        else:
            self.ml_classifier = None
            if use_ml_classifier and not ML_AVAILABLE:
                self.logger.warning("ML classifier requested but scikit-learn not available")
        
        # Initialize cost model
        self.cost_model = CostModel(
            commission_bps=1.0,
            slippage_bps=5.0,
            slippage_atr_multiplier=0.5
        )
        
        # Initialize Kelly sizing
        self.kelly_sizing = KellySizing(
            kelly_fraction=0.25,
            max_kelly_fraction=0.5,
            max_position_size=0.1,
            min_position_size=0.001
        )
        
        # Initialize risk guards
        self.risk_guards = RiskGuards(
            initial_capital=initial_capital,
            max_position_size=0.1,
            daily_loss_limit=0.03,
            weekly_loss_limit=0.08,
            max_drawdown_limit=0.15,
            max_positions=20
        )
        
        # Initialize liquidity guards
        self.liquidity_guards = LiquidityGuards(
            min_avg_volume=min_avg_volume,
            min_price=min_price,
            max_price=max_price,
            min_market_cap=100000000,    # $100M market cap (fixed)
            max_bid_ask_spread=max_bid_ask_spread,
            min_daily_volume=min_daily_volume
        )
        
        # Storage for trade plans
        self.trade_plans = []
        
        self.logger.info("OfficialTickerScanner with EVR framework initialized")
    
    def _load_trained_parameters(self) -> None:
        """Load trained parameters from historical backtesting if available."""
        try:
            from parameter_integration import integrate_trained_parameters
            
            params_path = Path("trained_parameters/scanner_parameters.json")
            
            # Check if parameters exist
            if params_path.exists():
                # Check age of parameters
                needs_retraining = self._check_parameters_age(params_path)
                
                if needs_retraining:
                    self.logger.info("⏰ Training parameters are 3+ days old, triggering retraining...")
                    self._retrain_parameters(is_initial=False)
            else:
                # Parameters don't exist - trigger initial training
                self.logger.info("📊 No trained parameters found, triggering initial training...")
                self._retrain_parameters(is_initial=True)
            
            # Load parameters (either existing or newly trained)
            if params_path.exists():
                success = integrate_trained_parameters(self, str(params_path))
                if success:
                    self.logger.info("✓ Loaded trained parameters from historical backtesting")
                else:
                    self.logger.warning("Failed to load trained parameters, using defaults")
            else:
                self.logger.warning("Training failed or parameters not created, using default priors")
                self.logger.info("You can manually train by running: python run_parameter_training.py")
                
        except ImportError:
            self.logger.warning("parameter_integration module not found, using default priors")
        except Exception as e:
            self.logger.warning(f"Error loading trained parameters: {e}, using defaults")
    
    def _check_parameters_age(self, params_path: Path) -> bool:
        """Check if training parameters are 3+ days old.
        
        Args:
            params_path: Path to parameters file
            
        Returns:
            True if parameters need retraining (3+ days old), False otherwise
        """
        try:
            # Read the parameters file
            with open(params_path, 'r') as f:
                params = json.load(f)
            
            # Get training date from metadata
            training_date_str = params.get('metadata', {}).get('training_date')
            if not training_date_str:
                self.logger.warning("No training date found in parameters, triggering retraining")
                return True
            
            # Parse training date
            training_date = datetime.fromisoformat(training_date_str)
            current_date = datetime.now()
            
            # Calculate age in days
            age_days = (current_date - training_date).days
            
            self.logger.info(f"Training parameters age: {age_days} days")
            
            # Return True if 3+ days old
            return age_days >= 3
            
        except Exception as e:
            self.logger.warning(f"Error checking parameter age: {e}, assuming retraining needed")
            return True
    
    def _retrain_parameters(self, is_initial: bool = False) -> None:
        """Retrain parameters by running the training script.
        
        Args:
            is_initial: True if this is initial training (no parameters exist), False for retraining
        """
        try:
            import subprocess
            
            if is_initial:
                self.logger.info("🔄 Starting initial parameter training...")
                self.console.print(Panel(
                    "[cyan]No trained parameters found - performing initial training[/cyan]\n"
                    "[yellow]Training probability models from historical data...[/yellow]\n"
                    "[blue]This may take 5-10 minutes. Please wait...[/blue]",
                    title="📊 Initial Training",
                    border_style="cyan"
                ))
            else:
                self.logger.info("🔄 Starting automatic parameter retraining...")
                self.console.print(Panel(
                    "[yellow]Training parameters are outdated (3+ days old)[/yellow]\n"
                    "[cyan]Automatically retraining parameters from historical data...[/cyan]\n"
                    "[blue]This may take 5-10 minutes. Please wait...[/blue]",
                    title="⏰ Automatic Retraining",
                    border_style="yellow"
                ))
            
            # Run the training script
            result = subprocess.run(
                [sys.executable, "run_parameter_training.py", "--mode", "train"],
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout
            )
            
            if result.returncode == 0:
                if is_initial:
                    self.logger.info("✓ Initial training completed successfully")
                    self.console.print("[green]✓ Parameters trained successfully![/green]")
                else:
                    self.logger.info("✓ Automatic retraining completed successfully")
                    self.console.print("[green]✓ Parameters retrained successfully![/green]")
            else:
                self.logger.error(f"Training failed with return code {result.returncode}")
                self.logger.error(f"Stderr: {result.stderr}")
                if is_initial:
                    self.console.print("[red]⚠ Initial training failed, using default parameters[/red]")
                else:
                    self.console.print("[red]⚠ Automatic retraining failed, using existing parameters[/red]")
                
        except subprocess.TimeoutExpired:
            self.logger.error("Training timed out after 15 minutes")
            self.console.print("[red]⚠ Training timed out, using default parameters[/red]")
        except FileNotFoundError:
            self.logger.error("run_parameter_training.py not found, skipping training")
            self.console.print("[red]⚠ Training script not found, using default parameters[/red]")
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            self.console.print(f"[red]⚠ Training error: {e}, using default parameters[/red]")
    
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
    
    def _get_cache_path(self, ticker: str, period: str) -> Path:
        """Get cache file path for a ticker.
        
        Args:
            ticker: Stock symbol
            period: Data period
            
        Returns:
            Path to cache file
        """
        cache_filename = f"{ticker}_{period}.parquet"
        return self.cache_dir / cache_filename
    
    def _load_from_cache(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Load ticker data from cache if available and fresh.
        
        Args:
            ticker: Stock symbol
            period: Data period
            
        Returns:
            DataFrame if cache exists and is fresh, None otherwise
        """
        cache_file = self._get_cache_path(ticker, period)
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is fresh (less than 24 hours old for 1y data, 6 hours for shorter periods)
            cache_age_hours = 24 if period == "1y" else 6
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            
            if cache_age < timedelta(hours=cache_age_hours):
                start_time = time.time()
                data = pd.read_parquet(cache_file)
                load_time = time.time() - start_time
                self.logger.debug(f"📂 Loaded {ticker} from cache in {load_time:.3f}s ({len(data)} rows)")
                return data
            else:
                self.logger.debug(f"⏰ Cache expired for {ticker} (age: {cache_age})")
                return None
        except Exception as e:
            self.logger.debug(f"Failed to load cache for {ticker}: {e}")
            return None
    
    def _save_to_cache(self, ticker: str, period: str, data: pd.DataFrame) -> None:
        """Save ticker data to cache.
        
        Args:
            ticker: Stock symbol
            period: Data period
            data: DataFrame to cache
        """
        if data is None or data.empty:
            return
        
        try:
            cache_file = self._get_cache_path(ticker, period)
            start_time = time.time()
            data.to_parquet(cache_file, compression='snappy')
            save_time = time.time() - start_time
            file_size = cache_file.stat().st_size / 1024  # KB
            self.logger.debug(f"💾 Cached {ticker} to {cache_file.name} in {save_time:.3f}s ({file_size:.1f}KB)")
        except Exception as e:
            self.logger.debug(f"❌ Failed to cache {ticker}: {e}")
    
    def _load_delisted_tickers(self) -> Set[str]:
        """Load delisted tickers from file.
        
        Returns:
            Set of delisted ticker symbols
        """
        try:
            if self.delisted_file.exists():
                with open(self.delisted_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('delisted_tickers', []))
            return set()
        except Exception as e:
            self.logger.debug(f"Failed to load delisted tickers: {e}")
            return set()
    
    def _save_delisted_tickers(self) -> None:
        """Save delisted tickers to file."""
        try:
            data = {
                'delisted_tickers': list(self.delisted_tickers),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.delisted_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f"Saved {len(self.delisted_tickers)} delisted tickers to {self.delisted_file}")
        except Exception as e:
            self.logger.debug(f"Failed to save delisted tickers: {e}")
    
    def _mark_ticker_delisted(self, ticker: str, reason: str = "No data available") -> None:
        """Mark a ticker as delisted.
        
        Args:
            ticker: Ticker symbol to mark as delisted
            reason: Reason for delisting
        """
        self.delisted_tickers.add(ticker)
        self._save_delisted_tickers()
        self.logger.debug(f"Marked {ticker} as delisted: {reason}")
    
    def _is_ticker_delisted(self, ticker: str) -> bool:
        """Check if a ticker is marked as delisted.
        
        Args:
            ticker: Ticker symbol to check
            
        Returns:
            True if ticker is delisted, False otherwise
        """
        return ticker in self.delisted_tickers
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob("*.parquet"))
            total_files = len(cache_files)
            
            if total_files == 0:
                return {"total_files": 0, "total_size_mb": 0, "oldest_file": None, "newest_file": None}
            
            total_size = sum(f.stat().st_size for f in cache_files)
            total_size_mb = total_size / (1024 * 1024)
            
            # Get file ages
            file_ages = [(f, datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)) for f in cache_files]
            oldest_file = min(file_ages, key=lambda x: x[1])[0].name
            newest_file = max(file_ages, key=lambda x: x[1])[0].name
            
            return {
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "oldest_file": oldest_file,
                "newest_file": newest_file,
                "delisted_tickers": len(self.delisted_tickers)
            }
        except Exception as e:
            self.logger.debug(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
    
    def get_stock_data(self, ticker: str, period: str = "1y", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Get stock data using caching and parallel fetcher with advanced rate limiting.
        
        Args:
            ticker: Stock symbol
            period: Data period (1y, 6mo, 3mo, etc.)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with stock data or None if failed
        """
        # Check if ticker is already marked as delisted
        if self._is_ticker_delisted(ticker):
            self.logger.debug(f"⏭️  Skipping delisted ticker: {ticker}")
            return None
        
        # Try cache first
        if use_cache:
            self.logger.debug(f"🔍 Checking cache for {ticker}...")
            cached_data = self._load_from_cache(ticker, period)
            if cached_data is not None:
                self.logger.debug(f"✅ Cache HIT for {ticker} ({len(cached_data)} rows)")
                return cached_data
            else:
                self.logger.debug(f"❌ Cache MISS for {ticker}")
        
        # Fetch from API
        self.logger.debug(f"🌐 Fetching {ticker} from API...")
        start_time = time.time()
        data = self.data_fetcher.fetch_ticker_data(ticker, period)
        fetch_time = time.time() - start_time
        
        if data is not None and not data.empty:
            self.logger.debug(f"✅ API SUCCESS for {ticker} ({len(data)} rows, {fetch_time:.2f}s)")
            # Save to cache for future use
            if use_cache:
                self.logger.debug(f"💾 Caching {ticker} data...")
                cache_start = time.time()
                self._save_to_cache(ticker, period, data)
                cache_time = time.time() - cache_start
                self.logger.debug(f"💾 Cached {ticker} in {cache_time:.2f}s")
            return data
        
        # Mark ticker as delisted if no data available
        self.logger.debug(f"❌ No data available for {ticker} after {fetch_time:.2f}s - marking as delisted")
        self._mark_ticker_delisted(ticker, "No data available from API")
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
            open_price = data['Open']
            volume = data['Volume']
            current_price = float(close.iloc[-1])
            
            # Moving averages
            indicators['sma_20'] = float(close.rolling(20).mean().iloc[-1])
            indicators['sma_50'] = float(close.rolling(50).mean().iloc[-1])
            indicators['ema_12'] = float(close.ewm(span=12).mean().iloc[-1])
            indicators['ema_26'] = float(close.ewm(span=26).mean().iloc[-1])
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rs_value = rs.iloc[-1]
            if pd.isna(rs_value) or rs_value == 0:
                indicators['rsi'] = 50.0  # Neutral RSI
            else:
                indicators['rsi'] = float(100 - (100 / (1 + rs_value)))
            
            # MACD - calculate from full EMA series for proper signal line
            ema_12_series = close.ewm(span=12).mean()
            ema_26_series = close.ewm(span=26).mean()
            macd_series = ema_12_series - ema_26_series
            signal_series = macd_series.ewm(span=9).mean()
            
            macd_line = float(macd_series.iloc[-1])
            signal_line = float(signal_series.iloc[-1])
            indicators['macd'] = macd_line
            indicators['macd_signal'] = signal_line
            indicators['macd_histogram'] = macd_line - signal_line
            
            # Bollinger Bands
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            indicators['bb_upper'] = float(bb_upper.iloc[-1])
            indicators['bb_lower'] = float(bb_lower.iloc[-1])
            
            # BB Position with division by zero protection
            bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            if bb_range > 0:
                indicators['bb_position'] = float((close.iloc[-1] - bb_lower.iloc[-1]) / bb_range)
            else:
                indicators['bb_position'] = 0.5  # Neutral position
            
            # Volume indicators
            indicators['volume_sma'] = float(volume.rolling(20).mean().iloc[-1])
            if indicators['volume_sma'] > 0:
                indicators['volume_ratio'] = float(volume.iloc[-1] / indicators['volume_sma'])
            else:
                indicators['volume_ratio'] = 1.0  # Default ratio
            
            # Price momentum
            indicators['momentum_5'] = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
            indicators['momentum_10'] = float((close.iloc[-1] / close.iloc[-11] - 1) * 100)
            
            # Volatility
            returns = close.pct_change()
            indicators['volatility_20'] = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
            
            # Support and Resistance levels
            high_20 = high.rolling(20).max()
            low_20 = low.rolling(20).min()
            indicators['resistance_level'] = float(high_20.iloc[-1])
            indicators['support_level'] = float(low_20.iloc[-1])
            indicators['resistance_distance'] = float((indicators['resistance_level'] - current_price) / current_price * 100)
            indicators['support_distance'] = float((current_price - indicators['support_level']) / current_price * 100)
            
            # Stochastic Oscillator
            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            stoch_range = highest_high - lowest_low
            k_percent = pd.Series(index=close.index, dtype=float)
            k_percent[stoch_range > 0] = 100 * ((close - lowest_low) / stoch_range)[stoch_range > 0]
            k_percent[stoch_range <= 0] = 50.0  # Neutral value
            indicators['stoch_k'] = float(k_percent.iloc[-1])
            indicators['stoch_d'] = float(k_percent.rolling(3).mean().iloc[-1])
            
            # Williams %R
            williams_range = highest_high.iloc[-1] - lowest_low.iloc[-1]
            if williams_range > 0:
                indicators['williams_r'] = float(-100 * ((highest_high.iloc[-1] - current_price) / williams_range))
            else:
                indicators['williams_r'] = -50.0  # Neutral value
            
            # Commodity Channel Index (CCI)
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            mad_value = mad.iloc[-1]
            if mad_value > 0:
                indicators['cci'] = float((typical_price.iloc[-1] - sma_tp.iloc[-1]) / (0.015 * mad_value))
            else:
                indicators['cci'] = 0.0  # Neutral CCI
            
            # Average True Range (ATR)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['atr'] = float(true_range.rolling(14).mean().iloc[-1])
            indicators['atr_percentage'] = float((indicators['atr'] / current_price) * 100)
            
            # Price patterns (simplified)
            high_low_range = high.iloc[-1] - low.iloc[-1]
            if high_low_range > 0:
                indicators['doji'] = bool(abs(close.iloc[-1] - open_price.iloc[-1]) / high_low_range < 0.1)
            else:
                indicators['doji'] = False
            indicators['hammer'] = bool((close.iloc[-1] > open_price.iloc[-1]) and ((close.iloc[-1] - low.iloc[-1]) > 2 * (high.iloc[-1] - close.iloc[-1])))
            indicators['engulfing'] = bool((close.iloc[-1] > open_price.iloc[-1]) and (close.iloc[-2] < open_price.iloc[-2]) and (open_price.iloc[-1] < close.iloc[-2]) and (close.iloc[-1] > open_price.iloc[-2]))
            
            # Trend strength
            indicators['adx'] = self._calculate_adx(high, low, close)
            
        except Exception as e:
            self.logger.debug(f"Error calculating indicators: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return {}
        
        return indicators
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)."""
        try:
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            dm_plus = high.diff()
            dm_minus = -low.diff()
            dm_plus[dm_plus < 0] = 0
            dm_minus[dm_minus < 0] = 0
            
            # Smoothed values
            tr_smooth = tr.ewm(alpha=1/period).mean()
            dm_plus_smooth = dm_plus.ewm(alpha=1/period).mean()
            dm_minus_smooth = dm_minus.ewm(alpha=1/period).mean()
            
            # Directional Indicators
            di_plus = 100 * (dm_plus_smooth / tr_smooth)
            di_minus = 100 * (dm_minus_smooth / tr_smooth)
            
            # ADX
            di_sum = di_plus + di_minus
            if di_sum.iloc[-1] > 0:
                dx = 100 * abs(di_plus - di_minus) / di_sum
            else:
                dx = pd.Series(0, index=di_plus.index)
            adx = dx.ewm(alpha=1/period).mean()
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        except:
            return 0
    
    def generate_signals(self, ticker: str, data: pd.DataFrame, indicators: Dict[str, float]) -> List[Signal]:
        """Generate trading signals based on technical indicators.
        
        Args:
            ticker: Stock symbol
            data: Stock price data
            indicators: Calculated technical indicators
            
        Returns:
            List of EVR Signal objects
        """
        signals = []
        
        if not indicators:
            return signals
        
        current_price = data['Close'].iloc[-1]
        timestamp = datetime.now()
        
        # Signal 1: Moving Average Crossover
        if indicators.get('sma_20', 0) > indicators.get('sma_50', 0):
            if current_price > indicators['sma_20']:
                signals.append(Signal(
                    symbol=ticker,
                    setup='ma_crossover',
                    direction=1,  # Long
                    strength=0.7,
                    timestamp=timestamp,
                    features=indicators.copy()
                ))
        
        # Signal 2: RSI Oversold/Overbought
        rsi = indicators.get('rsi', 50)
        if rsi < 30:  # Oversold
            signals.append(Signal(
                symbol=ticker,
                setup='rsi_oversold',
                direction=1,  # Long
                strength=0.6,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        elif rsi > 70:  # Overbought
            signals.append(Signal(
                symbol=ticker,
                setup='rsi_overbought',
                direction=-1,  # Short
                strength=0.6,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        
        # Signal 3: MACD Signal
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal and indicators.get('macd_histogram', 0) > 0:
            signals.append(Signal(
                symbol=ticker,
                setup='macd_bullish',
                direction=1,  # Long
                strength=0.65,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        
        # Signal 4: Bollinger Band Breakout
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position > 0.8:  # Near upper band
            signals.append(Signal(
                symbol=ticker,
                setup='bb_breakout',
                direction=1,  # Long
                strength=0.55,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        
        # Signal 5: Volume Confirmation
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:  # High volume
            momentum_5 = indicators.get('momentum_5', 0)
            if momentum_5 > 2:  # Positive momentum
                signals.append(Signal(
                    symbol=ticker,
                    setup='volume_momentum',
                    direction=1,  # Long
                    strength=0.6,
                    timestamp=timestamp,
                    features=indicators.copy()
                ))
        
        # Signal 6: Stochastic Oscillator
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        if stoch_k < 20 and stoch_d < 20:  # Oversold
            signals.append(Signal(
                symbol=ticker,
                setup='stoch_oversold',
                direction=1,  # Long
                strength=0.65,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        elif stoch_k > 80 and stoch_d > 80:  # Overbought
            signals.append(Signal(
                symbol=ticker,
                setup='stoch_overbought',
                direction=-1,  # Short
                strength=0.65,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        
        # Signal 7: Williams %R
        williams_r = indicators.get('williams_r', -50)
        if williams_r < -80:  # Oversold
            signals.append(Signal(
                symbol=ticker,
                setup='williams_oversold',
                direction=1,  # Long
                strength=0.62,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        elif williams_r > -20:  # Overbought
            signals.append(Signal(
                symbol=ticker,
                setup='williams_overbought',
                direction=-1,  # Short
                strength=0.62,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        
        # Signal 8: Commodity Channel Index (CCI)
        cci = indicators.get('cci', 0)
        if cci < -100:  # Oversold
            signals.append(Signal(
                symbol=ticker,
                setup='cci_oversold',
                direction=1,  # Long
                strength=0.68,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        elif cci > 100:  # Overbought
            signals.append(Signal(
                symbol=ticker,
                setup='cci_overbought',
                direction=-1,  # Short
                strength=0.68,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        
        # Signal 9: Support and Resistance Levels
        support_distance = indicators.get('support_distance', 100)
        resistance_distance = indicators.get('resistance_distance', 100)
        if support_distance < 2:  # Near support
            signals.append(Signal(
                symbol=ticker,
                setup='support_bounce',
                direction=1,  # Long
                strength=0.72,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        elif resistance_distance < 2:  # Near resistance
            signals.append(Signal(
                symbol=ticker,
                setup='resistance_rejection',
                direction=-1,  # Short
                strength=0.72,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        
        # Signal 10: Price Patterns
        if indicators.get('hammer', False):
            signals.append(Signal(
                symbol=ticker,
                setup='hammer_pattern',
                direction=1,  # Long
                strength=0.75,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        
        if indicators.get('engulfing', False):
            signals.append(Signal(
                symbol=ticker,
                setup='engulfing_pattern',
                direction=1,  # Long
                strength=0.78,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        
        if indicators.get('doji', False):
            signals.append(Signal(
                symbol=ticker,
                setup='doji_pattern',
                direction=1,  # Neutral signal, default to long
                strength=0.55,
                timestamp=timestamp,
                features=indicators.copy()
            ))
        
        # Signal 11: Trend Strength (ADX)
        adx = indicators.get('adx', 0)
        if adx > 25:  # Strong trend
            if indicators.get('sma_20', 0) > indicators.get('sma_50', 0):
                signals.append(Signal(
                    symbol=ticker,
                    setup='strong_uptrend',
                    direction=1,  # Long
                    strength=0.80,
                    timestamp=timestamp,
                    features=indicators.copy()
                ))
            else:
                signals.append(Signal(
                    symbol=ticker,
                    setup='strong_downtrend',
                    direction=-1,  # Short
                    strength=0.80,
                    timestamp=timestamp,
                    features=indicators.copy()
                ))
        
        # Signal 12: Volatility Breakout
        volatility = indicators.get('volatility_20', 0)
        atr_percentage = indicators.get('atr_percentage', 0)
        if volatility > 30 and atr_percentage > 2:  # High volatility
            momentum_5 = indicators.get('momentum_5', 0)
            if abs(momentum_5) > 5:  # Strong momentum
                direction = 1 if momentum_5 > 0 else -1
                signals.append(Signal(
                    symbol=ticker,
                    setup='volatility_breakout',
                    direction=direction,
                    strength=0.70,
                    timestamp=timestamp,
                    features=indicators.copy()
                ))
        
        return signals
    
    def _compute_r_unit_metrics(self, trade_plan: TradePlan) -> None:
        """Compute R-unit metrics for a trade plan.
        
        Args:
            trade_plan: TradePlan to compute metrics for
        """
        try:
            entry_price = trade_plan.entry
            stop_price = trade_plan.stop
            target_price = trade_plan.take_profit
            
            # Calculate R-unit values
            r_unit = abs(entry_price - stop_price)  # 1R = |Entry - Stop|
            if r_unit == 0:
                return
            
            # Determine direction: LONG if entry > stop, SHORT otherwise
            is_long = entry_price > stop_price
            
            # Calculate reward in R units (m)
            if is_long:
                # For long positions: reward = Target - Entry
                reward_r = abs(target_price - entry_price) / r_unit
            else:
                # For short positions: reward = Entry - Target
                reward_r = abs(entry_price - target_price) / r_unit
            
            # Calculate costs in R units
            # Total costs = commission + slippage (in dollars)
            total_costs_dollars = trade_plan.cost_bps * entry_price / 10000 + trade_plan.slippage_bps * entry_price / 10000
            costs_r = total_costs_dollars / r_unit
            
            # R-unit expectancy: E[R] = p·m - (1-p) - costs_R
            expectancy_r = trade_plan.p_win * reward_r - (1 - trade_plan.p_win) - costs_r
            
            # Calculate minimum win rate threshold
            min_win_rate = (1 + costs_r) / (reward_r + 1)
            
            # Safety margin: require 5-10 percentage points above minimum
            safety_margin = 0.075  # 7.5% safety margin
            required_win_rate = min_win_rate + safety_margin
            
            # Store R-unit metrics
            trade_plan.r_unit = r_unit
            trade_plan.reward_r = reward_r
            trade_plan.costs_r = costs_r
            trade_plan.expectancy_r = expectancy_r
            trade_plan.min_win_rate = min_win_rate
            trade_plan.required_win_rate = required_win_rate
            
        except Exception as e:
            self.logger.debug(f"Error computing R-unit metrics: {e}")
    
    def _assign_action(self, trade_plan: TradePlan) -> None:
        """Assign BUY/SHORT/NULL action based on expectancy and win rate.
        
        Args:
            trade_plan: TradePlan to assign action for
        """
        try:
            # Determine direction: LONG if entry > stop, SHORT otherwise
            is_long = trade_plan.entry > trade_plan.stop
            
            # Action only if expectancy > 0 and p_win >= required_win_rate
            if trade_plan.expectancy_r > 0 and trade_plan.p_win >= trade_plan.required_win_rate:
                trade_plan.action = "BUY" if is_long else "SHORT"
            else:
                trade_plan.action = "NULL"
                
        except Exception as e:
            self.logger.debug(f"Error assigning action: {e}")
            trade_plan.action = "NULL"
    
    def create_trade_plan(self, signal: Signal, data: pd.DataFrame) -> Optional[TradePlan]:
        """Create TradePlan from EVR Signal.
        
        Args:
            signal: EVR Signal object
            data: Stock price data
            
        Returns:
            TradePlan object or None if creation fails
        """
        try:
            current_price = float(data['Close'].iloc[-1])
            atr_percentage = signal.features.get('atr_percentage', 2.0)
            volatility = signal.features.get('volatility_20', 0.2)
            
            # Check liquidity requirements
            liquidity_checks = self.liquidity_guards.check_liquidity(data, current_price)
            if not liquidity_checks['all_checks_passed']:
                self.logger.debug(f"Liquidity check failed for {signal.symbol}: {liquidity_checks}")
                return None
            
            # Get probability estimates
            if self.use_ml_classifier and self.ml_classifier:
                p_win = self.ml_classifier.predict_probability(signal.features)
            else:
                p_win, avg_win, avg_loss = self.prob_model.estimate(
                    signal.setup, signal.symbol, volatility=volatility
                )
            
            self.logger.debug(f"Probability estimates for {signal.symbol}: p_win={p_win:.4f}, avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f}")
            
            # Calculate entry, stop, and targets
            if signal.direction == 1:  # Long
                entry = current_price
                stop = current_price * (1 - atr_percentage / 100 * 2)  # 2x ATR stop
                take_profit = current_price * (1 + atr_percentage / 100 * 4)  # 4x ATR target
                targets = [take_profit]
            else:  # Short
                entry = current_price
                stop = current_price * (1 + atr_percentage / 100 * 2)  # 2x ATR stop
                take_profit = current_price * (1 - atr_percentage / 100 * 4)  # 4x ATR target
                targets = [take_profit]
            
            # Calculate Kelly fraction
            kelly_fraction = self.kelly_sizing.calculate_kelly_fraction(p_win, avg_win, avg_loss)
            self.logger.debug(f"Kelly fraction for {signal.symbol}: {kelly_fraction:.4f}")
            
            # Calculate position size
            position_size, shares = self.kelly_sizing.size_position(
                self.risk_guards.current_capital, entry, stop, kelly_fraction
            )
            self.logger.debug(f"Position size for {signal.symbol}: ${position_size:.2f}, shares: {shares}")
            
            # Skip if position size is too small
            if position_size < 100:  # Minimum $100 position
                self.logger.debug(f"Position size too small for {signal.symbol}: ${position_size:.2f}")
                return None
            
            # Calculate costs
            commission_cost, slippage_cost = self.cost_model.calculate_costs(
                entry, atr_percentage, position_size
            )
            
            # Calculate expected return
            if signal.direction == 1:
                potential_return = (take_profit - entry) / entry
                potential_loss = (entry - stop) / entry
            else:
                potential_return = (entry - take_profit) / entry
                potential_loss = (stop - entry) / entry
            
            expected_return = p_win * potential_return - (1 - p_win) * potential_loss
            
            # Calculate risk in dollars
            risk_dollars = abs(entry - stop) * shares
            
            # Calculate cost basis points (avoid division by zero)
            total_cost = commission_cost + slippage_cost
            # Cost basis points should be based on the original cost model settings, not calculated from actual costs
            cost_bps = self.cost_model.commission_bps
            slippage_bps = self.cost_model.slippage_bps
            
            # Create TradePlan
            trade_plan = TradePlan(
                ticker=signal.symbol,
                setup=signal.setup,
                entry=entry,
                stop=stop,
                targets=targets,
                p_win=p_win,
                avg_r_win=avg_win,
                avg_r_loss=avg_loss,
                expected_return=expected_return,
                kelly_fraction=kelly_fraction,
                position_size=position_size,
                shares=shares,
                risk_dollars=risk_dollars,
                notes=f"Generated from {signal.setup} signal",
                signal_type=signal.setup,
                confidence=signal.strength,
                take_profit=take_profit,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps
            )
            
            # Compute R-unit metrics and assign action
            self._compute_r_unit_metrics(trade_plan)
            self._assign_action(trade_plan)
            
            # Calculate and store liquidity score
            trade_plan.liquidity_score = self.liquidity_guards.get_liquidity_score(data, current_price)
            
            return trade_plan
            
        except Exception as e:
            self.logger.debug(f"Failed to create trade plan for {signal.symbol}: {e}")
            return None
    
    
    def scan_tickers(self, tickers: List[str], max_tickers: int = 100, use_cache: bool = True) -> List[TradePlan]:
        """Scan tickers for trading signals.
        
        Args:
            tickers: List of ticker symbols
            max_tickers: Maximum number of tickers to scan (None for all tickers)
            use_cache: Whether to use cached data if available
            
        Returns:
            List of TradePlan objects
        """
        import random
        
        # Determine how many tickers to scan
        if max_tickers is None or max_tickers >= len(tickers):
            tickers_to_scan = len(tickers)
            processed_tickers = tickers
            self.logger.info(f"Starting scan of all {tickers_to_scan} tickers")
        else:
            tickers_to_scan = max_tickers
            # Use random sampling to avoid alphabetical bias
            processed_tickers = random.sample(tickers, tickers_to_scan)
            self.logger.info(f"Starting scan of {tickers_to_scan} randomly selected tickers from {len(tickers)} total")
        
        # Process tickers in parallel batches
        batch_size = min(10, tickers_to_scan)  # Process in batches of 10
        batches = [processed_tickers[i:i + batch_size] for i in range(0, len(processed_tickers), batch_size)]
        
        all_trade_plans = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Scanning tickers...", total=tickers_to_scan)
            
            for batch_idx, batch in enumerate(batches):
                batch_start = time.time()
                self.logger.debug(f"🔄 Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} tickers")
                
                # Process batch in parallel
                batch_results = self._process_ticker_batch(batch, use_cache=use_cache)
                all_trade_plans.extend(batch_results)
                
                batch_time = time.time() - batch_start
                self.logger.debug(f"✅ Batch {batch_idx + 1} completed in {batch_time:.2f}s ({len(batch_results)} trade plans)")
                
                # Update progress
                progress.advance(task, len(batch))
                
                # Minimal delay between batches (reduced from 0.5s to 0.1s for speed)
                if len(batches) > 1:
                    time.sleep(0.1)
        
        # Rank by R-unit expectancy: E[R] = p·m - (1-p) - costs_R
        def r_unit_expectancy_score(trade_plan: TradePlan) -> float:
            """Calculate R-unit expectancy score for ranking."""
            try:
                # Extract trade plan parameters
                entry_price = trade_plan.entry
                stop_price = trade_plan.stop
                target_price = trade_plan.take_profit
                p_win = trade_plan.p_win
                
                # Calculate R-unit values
                r_unit = abs(entry_price - stop_price)  # 1R = |Entry - Stop|
                if r_unit == 0:
                    return 0.0
                
                # Calculate reward in R units (m)
                if trade_plan.signal_type and 'SHORT' in str(trade_plan.signal_type):
                    # For short positions: reward = Stop - Target
                    reward_r = abs(stop_price - target_price) / r_unit
                else:
                    # For long positions: reward = Target - Entry
                    reward_r = abs(target_price - entry_price) / r_unit
                
                # Calculate costs in R units
                # Total costs = commission + slippage (in dollars)
                total_costs_dollars = trade_plan.cost_bps * entry_price / 10000 + trade_plan.slippage_bps * entry_price / 10000
                costs_r = total_costs_dollars / r_unit
                
                # R-unit expectancy: E[R] = p·m - (1-p) - costs_R
                expectancy_r = p_win * reward_r - (1 - p_win) - costs_r
                
                # Calculate minimum win rate threshold
                min_win_rate = (1 + costs_r) / (reward_r + 1)
                
                # Safety margin: require 5-10 percentage points above minimum
                safety_margin = 0.075  # 7.5% safety margin
                required_win_rate = min_win_rate + safety_margin
                
                # Bonus for meeting safety margin
                safety_bonus = 0.0
                if p_win >= required_win_rate:
                    safety_bonus = 0.1  # 10% bonus for meeting safety margin
                
                # Final score: expectancy + safety bonus
                final_score = expectancy_r + safety_bonus
                
                # Store R-unit metrics in trade plan for display
                trade_plan.r_unit = r_unit
                trade_plan.reward_r = reward_r
                trade_plan.costs_r = costs_r
                trade_plan.expectancy_r = expectancy_r
                trade_plan.min_win_rate = min_win_rate
                trade_plan.required_win_rate = required_win_rate
                
                return final_score
                
            except Exception as e:
                self.logger.debug(f"Error calculating R-unit expectancy score: {e}")
                return 0.0
        
        # Sort by R-unit expectancy score (descending)
        all_trade_plans.sort(key=r_unit_expectancy_score, reverse=True)
        
        self.logger.info(f"Scan completed: {len(all_trade_plans)} trade plans from {tickers_to_scan} tickers")
        return all_trade_plans
    
    def _process_ticker_batch(self, tickers: List[str], use_cache: bool = True) -> List[TradePlan]:
        """Process a batch of tickers in parallel.
        
        Args:
            tickers: List of ticker symbols to process
            use_cache: Whether to use cached data if available
            
        Returns:
            List of TradePlan objects
        """
        if not tickers:
            self.logger.debug("No tickers provided for processing; skipping batch")
            return []

        def process_single_ticker(ticker: str) -> List[TradePlan]:
            """Process a single ticker and return trade plans."""
            ticker_start = time.time()
            try:
                # Get stock data
                data = self.get_stock_data(ticker, use_cache=use_cache)
                if data is None or len(data) < 50:
                    self.logger.debug(f"⏭️  {ticker}: No data or insufficient data ({len(data) if data is not None else 0} rows)")
                    return []
                
                # Calculate indicators
                indicators_start = time.time()
                indicators = self.calculate_technical_indicators(data)
                indicators_time = time.time() - indicators_start
                if not indicators:
                    self.logger.debug(f"⏭️  {ticker}: No indicators calculated")
                    return []
                
                # Generate signals
                signals_start = time.time()
                signals = self.generate_signals(ticker, data, indicators)
                signals_time = time.time() - signals_start
                
                # Convert signals to trade plans
                trade_plans = []
                for signal in signals:
                    trade_plan = self.create_trade_plan(signal, data)
                    if trade_plan is not None:
                        # Check risk guards
                        risk_checks = self.risk_guards.check_trade_plan(trade_plan)
                        if risk_checks['all_checks_passed']:
                            trade_plans.append(trade_plan)
                
                total_time = time.time() - ticker_start
                self.logger.debug(f"✅ {ticker}: {len(trade_plans)} trade plans in {total_time:.2f}s (indicators: {indicators_time:.2f}s, signals: {signals_time:.2f}s)")
                return trade_plans
                
            except Exception as e:
                total_time = time.time() - ticker_start
                self.logger.debug(f"❌ {ticker}: Error after {total_time:.2f}s - {e}")
                return []
        
        # Process tickers in parallel
        all_trade_plans = []
        
        max_workers = min(12, len(tickers))
        if max_workers <= 0:
            return []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(process_single_ticker, ticker): ticker 
                for ticker in tickers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    trade_plans = future.result()
                    all_trade_plans.extend(trade_plans)
                except Exception as e:
                    self.logger.debug(f"Failed to process {ticker}: {e}")
        
        return all_trade_plans
    
    def _process_backtest_batch(self, tickers: List[str], date: str) -> List[Dict[str, Any]]:
        """Process a batch of tickers for backtesting in parallel.
        
        Args:
            tickers: List of ticker symbols to process
            date: Date to get recommendations for
            
        Returns:
            List of recommendations
        """
        if not tickers:
            self.logger.debug("No tickers provided for backtest processing; skipping batch")
            return []

        def process_single_backtest_ticker(ticker: str) -> List[Dict[str, Any]]:
            """Process a single ticker for backtesting."""
            try:
                # Get historical data up to the specific date
                import yfinance as yf
                
                # Calculate start date to ensure we have enough data for indicators
                start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
                
                # Get historical data up to the backtest date
                hist_data = None
                for attempt in range(2):  # Reduced attempts for parallel processing
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        hist_data = ticker_obj.history(start=start_date, end=date)
                        if not hist_data.empty:
                            break
                    except Exception as e:
                        if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                            wait_time = (2 ** attempt) * 1.5 + random.uniform(0.5, 1.5)
                            time.sleep(wait_time)
                            continue
                        else:
                            break
                
                if hist_data is None or hist_data.empty or len(hist_data) < 50:
                    return []
                
                # Calculate indicators
                indicators = self.calculate_technical_indicators(hist_data)
                if not indicators:
                    return []
                
                # Generate signals
                signals = self.generate_signals(ticker, hist_data, indicators)
                if not signals:
                    return []
                
                # Create trade plans
                trade_plans = []
                for signal in signals:
                    trade_plan = self.create_trade_plan(signal, hist_data)
                    if trade_plan:
                        trade_plans.append(trade_plan)
                
                if trade_plans:
                    # Aggregate trade plans for this ticker
                    aggregated = self.aggregate_trade_plans(trade_plans)
                    return aggregated
                
                return []
                
            except Exception as e:
                self.logger.debug(f"Error processing {ticker} for {date}: {e}")
                return []
        
        # Process tickers in parallel
        all_recommendations = []
        
        max_workers = min(8, len(tickers))
        if max_workers <= 0:
            return []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(process_single_backtest_ticker, ticker): ticker 
                for ticker in tickers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    recommendations = future.result()
                    all_recommendations.extend(recommendations)
                except Exception as e:
                    self.logger.debug(f"Failed to process {ticker} for {date}: {e}")
        
        return all_recommendations
        
        # Rank by R-unit expectancy: E[R] = p·m - (1-p) - costs_R
        def r_unit_expectancy_score(trade_plan: TradePlan) -> float:
            """Calculate R-unit expectancy score for ranking."""
            try:
                # Extract trade plan parameters
                entry_price = trade_plan.entry
                stop_price = trade_plan.stop
                target_price = trade_plan.take_profit
                p_win = trade_plan.p_win
                
                # Calculate R-unit values
                r_unit = abs(entry_price - stop_price)  # 1R = |Entry - Stop|
                if r_unit == 0:
                    return 0.0
                
                # Calculate reward in R units (m)
                if trade_plan.signal_type and 'SHORT' in str(trade_plan.signal_type):
                    # For short positions: reward = Stop - Target
                    reward_r = abs(stop_price - target_price) / r_unit
                else:
                    # For long positions: reward = Target - Entry
                    reward_r = abs(target_price - entry_price) / r_unit
                
                # Calculate costs in R units
                # Total costs = commission + slippage (in dollars)
                total_costs_dollars = trade_plan.cost_bps * entry_price / 10000 + trade_plan.slippage_bps * entry_price / 10000
                costs_r = total_costs_dollars / r_unit
                
                # R-unit expectancy: E[R] = p·m - (1-p) - costs_R
                expectancy_r = p_win * reward_r - (1 - p_win) - costs_r
                
                # Calculate minimum win rate threshold
                min_win_rate = (1 + costs_r) / (reward_r + 1)
                
                # Safety margin: require 5-10 percentage points above minimum
                safety_margin = 0.075  # 7.5% safety margin
                required_win_rate = min_win_rate + safety_margin
                
                # Bonus for meeting safety margin
                safety_bonus = 0.0
                if p_win >= required_win_rate:
                    safety_bonus = 0.1  # 10% bonus for meeting safety margin
                
                # Final score: expectancy + safety bonus
                final_score = expectancy_r + safety_bonus
                
                # Store R-unit metrics in trade plan for display
                trade_plan.r_unit = r_unit
                trade_plan.reward_r = reward_r
                trade_plan.costs_r = costs_r
                trade_plan.expectancy_r = expectancy_r
                trade_plan.min_win_rate = min_win_rate
                trade_plan.required_win_rate = required_win_rate
                
                return final_score
                
            except Exception as e:
                self.logger.debug(f"Error calculating R-unit expectancy score: {e}")
                return 0.0
        
        # Sort by R-unit expectancy score (descending)
        all_trade_plans.sort(key=r_unit_expectancy_score, reverse=True)
        
        self.logger.info(f"Scan completed: {len(all_trade_plans)} trade plans from {tickers_to_scan} tickers")
        return all_trade_plans
    
    def display_results(self, trade_plans: List[TradePlan], top_n: int = 20) -> None:
        """Display EVR TradePlan results in a formatted table.
        
        Args:
            trade_plans: List of TradePlan objects
            top_n: Number of top trade plans to display
        """
        if not trade_plans:
            self.console.print("[red]No trade plans found[/red]")
            return
        
        # Display EVR TradePlan results with R-unit metrics
        table = Table(title=f"Top {min(top_n, len(trade_plans))} EVR Trade Plans (R-Unit Ranking)")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Ticker", style="magenta", width=8)
        table.add_column("Setup", style="blue", width=15)
        table.add_column("Direction", style="green", width=8)
        table.add_column("Action", style="yellow", width=8)
        table.add_column("Entry", style="yellow", width=10)
        table.add_column("Stop", style="red", width=10)
        table.add_column("Target", style="green", width=10)
        table.add_column("P(Win)", style="cyan", width=8)
        table.add_column("Reward", style="green", width=8)
        table.add_column("Costs", style="red", width=8)
        table.add_column("E[R]", style="blue", width=8)
        table.add_column("Min P", style="yellow", width=8)
        table.add_column("Liquidity", style="cyan", width=8)
        table.add_column("Score", style="magenta", width=8)
        
        for i, plan in enumerate(trade_plans[:top_n], 1):
            # LONG: stop < entry (stop below entry, buy to profit from upward move)
            # SHORT: stop > entry (stop above entry, sell to profit from downward move)
            direction = "LONG" if plan.entry > plan.stop else "SHORT"
            # Color code based on safety margin
            p_win_color = "green" if plan.p_win >= plan.required_win_rate else "red"
            expectancy_color = "green" if plan.expectancy_r > 0 else "red"
            
            # Color code action: green for BUY/SHORT, yellow for NULL
            action_color = "green" if plan.action in ["BUY", "SHORT"] else "yellow"
            
            table.add_row(
                str(i),
                plan.ticker,
                plan.setup,
                direction,
                f"[{action_color}]{plan.action}[/{action_color}]",
                f"${plan.entry:.2f}",
                f"${plan.stop:.2f}",
                f"${plan.targets[0]:.2f}",
                f"[{p_win_color}]{plan.p_win:.1%}[/{p_win_color}]",
                f"{plan.reward_r:.2f}R",
                f"{plan.costs_r:.2f}R",
                f"[{expectancy_color}]{plan.expectancy_r:.3f}R[/{expectancy_color}]",
                f"{plan.min_win_rate:.1%}",
                f"{plan.liquidity_score:.2f}",
                f"{plan.expectancy_r + (0.1 if plan.p_win >= plan.required_win_rate else 0):.3f}"
            )
        
        self.console.print(table)
        
        # Display summary statistics
        total_plans = len(trade_plans)
        # LONG: stop < entry (stop below entry)
        # SHORT: stop > entry (stop above entry)
        long_plans = len([p for p in trade_plans if p.entry > p.stop])
        short_plans = len([p for p in trade_plans if p.entry < p.stop])
        
        avg_p_win = sum(p.p_win for p in trade_plans) / total_plans
        avg_expected_return = sum(p.expected_return for p in trade_plans) / total_plans
        avg_kelly = sum(p.kelly_fraction for p in trade_plans) / total_plans
        total_risk = sum(p.risk_dollars for p in trade_plans)
        
        summary_text = f"""
EVR Trade Plan Summary:
  Total Plans: {total_plans}
  Long Positions: {long_plans} ({long_plans/total_plans:.1%})
  Short Positions: {short_plans} ({short_plans/total_plans:.1%})
  Average P(Win): {avg_p_win:.1%}
  Average Expected Return: {avg_expected_return:.2%}
  Average Kelly Fraction: {avg_kelly:.1%}
  Total Risk: ${total_risk:,.0f}
  Available Capital: ${self.risk_guards.current_capital:,.0f}
        """
        
        panel = Panel(summary_text, title="EVR Scan Results", border_style="green")
        self.console.print(panel)
    
    def save_results(self, trade_plans: List[TradePlan], filename_prefix: str = "evr_trade_plans") -> None:
        """Save EVR TradePlan results to files.
        
        Args:
            trade_plans: List of TradePlan objects
            filename_prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert TradePlan objects to dictionaries for JSON/CSV export
        # Filter out NULL actions to focus on actionable recommendations
        actionable_plans = [plan for plan in trade_plans if plan.action != "NULL"]
        trade_plan_dicts = []
        for plan in actionable_plans:
            plan_dict = {
                'ticker': plan.ticker,
                'setup': plan.setup,
                'entry': plan.entry,
                'stop': plan.stop,
                'targets': plan.targets,
                'p_win': plan.p_win,
                'avg_r_win': plan.avg_r_win,
                'avg_r_loss': plan.avg_r_loss,
                'expected_return': plan.expected_return,
                'kelly_fraction': plan.kelly_fraction,
                'position_size': plan.position_size,
                'risk_dollars': plan.risk_dollars,
                'notes': plan.notes,
                'signal_type': plan.signal_type,
                'confidence': plan.confidence,
                'take_profit': plan.take_profit,
                'cost_bps': plan.cost_bps,
                'slippage_bps': plan.slippage_bps,
                'r_unit': plan.r_unit,
                'reward_r': plan.reward_r,
                'costs_r': plan.costs_r,
                'expectancy_r': plan.expectancy_r,
                'min_win_rate': plan.min_win_rate,
                'required_win_rate': plan.required_win_rate,
                'action': plan.action,
                'liquidity_score': plan.liquidity_score
            }
            trade_plan_dicts.append(plan_dict)
        
        # Save as CSV
        csv_file = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
        df = pd.DataFrame(trade_plan_dicts)
        df.to_csv(csv_file, index=False)
        self.logger.info(f"Saved {len(actionable_plans)} actionable trade plans to {csv_file}")
        
        # Save as JSON
        json_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(trade_plan_dicts, f, indent=2, default=str)
        self.logger.info(f"Saved detailed trade plans to {json_file}")
        
        # Save summary
        summary_file = self.output_dir / f"{filename_prefix}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"EVR Trade Plans Summary (Actionable Only)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Trade Plans: {len(trade_plans)} (All)\n")
            f.write(f"Actionable Plans: {len(actionable_plans)} (BUY/SHORT only)\n")
            f.write(f"Filtered Out: {len(trade_plans) - len(actionable_plans)} (NULL actions)\n\n")
            
            # Calculate statistics for actionable plans only
            total_plans = len(actionable_plans)
            if total_plans == 0:
                f.write("No actionable recommendations found.\n")
                return
            # LONG: stop < entry (stop below entry)
            # SHORT: stop > entry (stop above entry)
            long_plans = len([p for p in actionable_plans if p.entry > p.stop])
            short_plans = len([p for p in actionable_plans if p.entry < p.stop])
            buy_actions = len([p for p in actionable_plans if p.action == "BUY"])
            short_actions = len([p for p in actionable_plans if p.action == "SHORT"])
            avg_p_win = sum(p.p_win for p in actionable_plans) / total_plans
            avg_expected_return = sum(p.expected_return for p in actionable_plans) / total_plans
            avg_kelly = sum(p.kelly_fraction for p in actionable_plans) / total_plans
            total_risk = sum(p.risk_dollars for p in actionable_plans)
            
            f.write(f"Summary Statistics (Actionable Only):\n")
            f.write(f"  Long Positions: {long_plans} ({long_plans/total_plans:.1%})\n")
            f.write(f"  Short Positions: {short_plans} ({short_plans/total_plans:.1%})\n")
            f.write(f"  BUY Actions: {buy_actions} ({buy_actions/total_plans:.1%})\n")
            f.write(f"  SHORT Actions: {short_actions} ({short_actions/total_plans:.1%})\n")
            f.write(f"  Average P(Win): {avg_p_win:.1%}\n")
            f.write(f"  Average Expected Return: {avg_expected_return:.2%}\n")
            f.write(f"  Average Kelly Fraction: {avg_kelly:.1%}\n")
            f.write(f"  Total Risk: ${total_risk:,.0f}\n")
            f.write(f"  Available Capital: ${self.risk_guards.current_capital:,.0f}\n\n")
            
            f.write(f"Top 15 Actionable Trade Plans:\n")
            for i, plan in enumerate(actionable_plans[:15], 1):
                # LONG: stop < entry (stop below entry, buy to profit from upward move)
                # SHORT: stop > entry (stop above entry, sell to profit from downward move)
                direction = "LONG" if plan.entry > plan.stop else "SHORT"
                f.write(f"{i:2d}. {plan.ticker:6s} {plan.setup:15s} {direction:5s} {plan.action:5s} "
                       f"P(Win): {plan.p_win:5.1%} E[R]: {plan.expected_return:6.2%} "
                       f"Kelly: {plan.kelly_fraction:5.1%} Risk: ${plan.risk_dollars:8,.0f}\n")
        
        self.logger.info(f"Saved summary to {summary_file}")
    
    def aggregate_trade_plans(self, trade_plans: List[TradePlan]) -> List[Dict[str, Any]]:
        """Aggregate multiple trade plans per ticker into unified EVR recommendations.
        
        Args:
            trade_plans: List of EVR TradePlan objects
            
        Returns:
            List of aggregated ticker recommendations with EVR metrics
        """
        # Group trade plans by ticker
        ticker_plans = {}
        for plan in trade_plans:
            ticker = plan.ticker
            if ticker not in ticker_plans:
                ticker_plans[ticker] = []
            ticker_plans[ticker].append(plan)
        
        aggregated = []
        
        for ticker, plans in ticker_plans.items():
            if not plans:
                continue

            # Calculate aggregated EVR metrics
            total_plans = len(plans)

            # Separate plans by direction for primary aggregation
            long_plans = [plan for plan in plans if plan.entry > plan.stop]
            short_plans = [plan for plan in plans if plan.entry < plan.stop]

            # Determine primary direction (default to LONG on ties to keep behaviour stable)
            if len(long_plans) >= len(short_plans):
                primary_direction = "LONG"
                primary_plans = long_plans
                opposing_direction = "SHORT"
                opposing_plans = short_plans
            else:
                primary_direction = "SHORT"
                primary_plans = short_plans
                opposing_direction = "LONG"
                opposing_plans = long_plans

            # Fallback to all plans if we somehow have no plans for the primary side
            if not primary_plans:
                primary_plans = plans
                primary_direction = "LONG" if plans[0].entry > plans[0].stop else "SHORT"
                opposing_direction = "SHORT" if primary_direction == "LONG" else "LONG"
                opposing_plans = []

            primary_plan_count = len(primary_plans)
            opposing_plan_count = len(opposing_plans)

            # FIX: Use weighted averages based on the primary-direction plans only
            total_weight = sum(abs(plan.expected_return) for plan in primary_plans)

            if total_weight > 0:
                weighted_p_win = sum(plan.p_win * abs(plan.expected_return) for plan in primary_plans) / total_weight
                weighted_expected_return = sum(plan.expected_return * abs(plan.expected_return) for plan in primary_plans) / total_weight
                weighted_kelly = sum(plan.kelly_fraction * abs(plan.expected_return) for plan in primary_plans) / total_weight
            else:
                weighted_p_win = (
                    sum(plan.p_win for plan in primary_plans) / primary_plan_count
                    if primary_plan_count else 0.0
                )
                weighted_expected_return = (
                    sum(plan.expected_return for plan in primary_plans) / primary_plan_count
                    if primary_plan_count else 0.0
                )
                weighted_kelly = (
                    sum(plan.kelly_fraction for plan in primary_plans) / primary_plan_count
                    if primary_plan_count else 0.0
                )

            # Get the best plan within the primary direction (fallback to any plan if needed)
            if primary_plans:
                best_plan = max(primary_plans, key=lambda p: p.kelly_fraction * p.expected_return)
            else:
                best_plan = max(plans, key=lambda p: p.kelly_fraction * p.expected_return)

            # Calculate signal diversity and consensus using primary-direction plans
            primary_signal_types = {}
            for plan in primary_plans:
                primary_signal_types[plan.setup] = primary_signal_types.get(plan.setup, 0) + 1
            diversity_score = min(len(primary_signal_types) / 12.0, 1.0) if primary_plans else 0.0
            consensus_score = min(primary_plan_count / total_plans, 1.0) if total_plans else 0.0

            # Calculate EVR composite score
            evr_composite_score = self._calculate_evr_composite_score(
                weighted_p_win,
                weighted_expected_return,
                weighted_kelly,
                diversity_score,
                consensus_score,
                best_plan
            )

            # Calculate aggregated risk metrics (report deployable risk and directional context)
            direction_risk_total = sum(plan.risk_dollars for plan in primary_plans)
            opposing_risk_total = sum(plan.risk_dollars for plan in opposing_plans)
            deployable_risk = best_plan.risk_dollars if best_plan else 0.0
            avg_cost_bps = (
                sum(plan.cost_bps for plan in primary_plans) / primary_plan_count
                if primary_plan_count else 0.0
            )
            avg_slippage_bps = (
                sum(plan.slippage_bps for plan in primary_plans) / primary_plan_count
                if primary_plan_count else 0.0
            )

            # Get signal type breakdowns for transparency
            all_signal_type_counts = {}
            for plan in plans:
                signal_type = plan.setup
                all_signal_type_counts[signal_type] = all_signal_type_counts.get(signal_type, 0) + 1

            primary_setups_summary = (
                ', '.join([f"{k}({v})" for k, v in primary_signal_types.items()])
                if primary_signal_types else 'n/a'
            )
            overall_setups_summary = (
                ', '.join([f"{k}({v})" for k, v in all_signal_type_counts.items()])
                if all_signal_type_counts else 'n/a'
            )

            # Compute aggregated action based on best plan's expectancy and required win rate
            if best_plan.expectancy_r > 0 and weighted_p_win >= best_plan.required_win_rate:
                aggregated_action = "BUY" if primary_direction == "LONG" else "SHORT"
            else:
                aggregated_action = "NULL"

            # Create aggregated recommendation
            aggregated_recommendation = {
                'ticker': ticker,
                'total_signals': total_plans,
                'primary_direction_signals': primary_plan_count,
                'opposing_direction_signals': opposing_plan_count,
                'signal_types': all_signal_type_counts,
                'primary_signal_types': primary_signal_types,
                'primary_direction': primary_direction,
                'action': aggregated_action,
                'entry_price': best_plan.entry,
                'stop_loss': best_plan.stop,
                'take_profit': best_plan.take_profit,
                'targets': best_plan.targets,

                # EVR aggregated metrics
                'weighted_p_win': weighted_p_win,
                'weighted_expected_return': weighted_expected_return,
                'weighted_kelly_fraction': weighted_kelly,
                'total_risk_dollars': deployable_risk,
                'direction_risk_dollars': direction_risk_total,
                'opposing_direction_risk_dollars': opposing_risk_total,
                'avg_cost_bps': avg_cost_bps,
                'avg_slippage_bps': avg_slippage_bps,

                # Composite scoring
                'diversity_score': diversity_score,
                'consensus_score': consensus_score,
                'evr_composite_score': evr_composite_score,

                # Best plan details
                'best_setup': best_plan.setup,
                'best_confidence': best_plan.confidence,
                'best_position_size': best_plan.position_size,

                # Summary
                'signal_summary': (
                    f"{total_plans} signals (primary {primary_direction}: {primary_plan_count}, "
                    f"{opposing_direction}: {opposing_plan_count}) | "
                    f"Primary setups: {primary_setups_summary} | "
                    f"All setups: {overall_setups_summary}"
                )
            }

            aggregated.append(aggregated_recommendation)
        
        # Sort by EVR composite score (highest first)
        aggregated.sort(key=lambda x: x['evr_composite_score'], reverse=True)
        
        return aggregated
    
    def _calculate_evr_composite_score(self, p_win: float, expected_return: float, kelly_fraction: float,
                                     diversity_score: float, consensus_score: float, best_plan: TradePlan) -> float:
        """Calculate EVR composite score using R-unit expectancy for aggregated recommendations.
        
        Args:
            p_win: Weighted probability of winning
            expected_return: Weighted expected return
            kelly_fraction: Weighted Kelly fraction
            diversity_score: Signal diversity score
            consensus_score: Direction consensus score
            best_plan: Best individual trade plan
            
        Returns:
            EVR composite score based on R-unit expectancy
        """
        try:
            # Clamp normalized components to 0-1 before combining
            diversity_score = max(0.0, min(diversity_score, 1.0))
            consensus_score = max(0.0, min(consensus_score, 1.0))

            # FIX: Properly normalize all components before combining
            # Use R-unit expectancy from the best plan
            if hasattr(best_plan, 'expectancy_r') and best_plan.expectancy_r is not None:
                expectancy_r = best_plan.expectancy_r
                
                # Normalize expectancy_r (typical range -1.0 to 3.0) to 0-1 scale
                normalized_expectancy = max(0, min((expectancy_r + 1.0) / 4.0, 1.0))
                
                # Safety margin bonus (0 or 0.2 contribution)
                safety_bonus = 0.0
                if hasattr(best_plan, 'required_win_rate') and p_win >= best_plan.required_win_rate:
                    safety_bonus = 0.2
                
                # FIX: Properly weight normalized components
                # Expectancy is most important (40%), safety margin (20%), diversity (20%), consensus (20%)
                composite_score = (
                    normalized_expectancy * 0.40 +
                    safety_bonus +
                    diversity_score * 0.20 +
                    consensus_score * 0.20
                )
                
                return max(0.0, min(composite_score, 1.0))
            else:
                # Fallback to original scoring if R-unit metrics not available
                probability_score = min(p_win, 1.0)
                payoff_score = min(abs(expected_return) / 0.20, 1.0)
                kelly_score = min(kelly_fraction / 0.25, 1.0)
                
                evr_score = (
                    probability_score * 0.25 +
                    payoff_score * 0.20 +
                    kelly_score * 0.15 +
                    diversity_score * 0.15 +
                    consensus_score * 0.10 +
                    min(1.0 / (1.0 + best_plan.risk_dollars / 1000), 1.0) * 0.10 +
                    min(1.0 / (1.0 + best_plan.cost_bps / 100), 1.0) * 0.05
                )
                
                return min(max(evr_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.debug(f"Error calculating EVR composite score: {e}")
            return 0.0
    
    def display_aggregated_results(self, aggregated_recommendations: List[Dict[str, Any]], top_n: int = 20) -> None:
        """Display aggregated EVR recommendations.
        
        Args:
            aggregated_recommendations: List of aggregated recommendations
            top_n: Number of top recommendations to display
        """
        if not aggregated_recommendations:
            self.console.print("[red]No aggregated recommendations found[/red]")
            return
        
        # Display aggregated EVR results
        table = Table(title=f"Top {min(top_n, len(aggregated_recommendations))} EVR Aggregated Recommendations")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Ticker", style="magenta", width=8)
        table.add_column("Signals", style="blue", width=8)
        table.add_column("Direction", style="green", width=8)
        table.add_column("Action", style="yellow", width=8)
        table.add_column("Entry", style="yellow", width=10)
        table.add_column("Stop", style="red", width=10)
        table.add_column("Target", style="green", width=10)
        table.add_column("P(Win)", style="cyan", width=8)
        table.add_column("E[R]", style="green", width=8)
        table.add_column("Kelly", style="blue", width=8)
        table.add_column("EVR Score", style="magenta", width=10)
        table.add_column("Risk $", style="red", width=10)
        
        for i, rec in enumerate(aggregated_recommendations[:top_n], 1):
            # Color code action: green for BUY/SHORT, yellow for NULL
            action_color = "green" if rec.get('action', 'NULL') in ["BUY", "SHORT"] else "yellow"
            action = rec.get('action', 'NULL')
            
            table.add_row(
                str(i),
                rec['ticker'],
                str(rec['total_signals']),
                rec['primary_direction'],
                f"[{action_color}]{action}[/{action_color}]",
                f"${rec['entry_price']:.2f}",
                f"${rec['stop_loss']:.2f}",
                f"${rec['take_profit']:.2f}",
                f"{rec['weighted_p_win']:.1%}",
                f"{rec['weighted_expected_return']:.1%}",
                f"{rec['weighted_kelly_fraction']:.1%}",
                f"{rec['evr_composite_score']:.3f}",
                f"${rec['total_risk_dollars']:,.0f}"
            )
        
        self.console.print(table)
        
        # Display summary statistics
        total_recommendations = len(aggregated_recommendations)
        long_recommendations = len([r for r in aggregated_recommendations if r['primary_direction'] == 'LONG'])
        short_recommendations = len([r for r in aggregated_recommendations if r['primary_direction'] == 'SHORT'])
        buy_actions = len([r for r in aggregated_recommendations if r.get('action') == 'BUY'])
        short_actions = len([r for r in aggregated_recommendations if r.get('action') == 'SHORT'])
        null_actions = len([r for r in aggregated_recommendations if r.get('action') == 'NULL'])
        
        avg_p_win = sum(r['weighted_p_win'] for r in aggregated_recommendations) / total_recommendations
        avg_expected_return = sum(r['weighted_expected_return'] for r in aggregated_recommendations) / total_recommendations
        avg_kelly = sum(r['weighted_kelly_fraction'] for r in aggregated_recommendations) / total_recommendations
        avg_evr_score = sum(r['evr_composite_score'] for r in aggregated_recommendations) / total_recommendations
        total_risk = sum(r['total_risk_dollars'] for r in aggregated_recommendations)
        
        summary_text = f"""
EVR Aggregated Summary:
  Total Recommendations: {total_recommendations}
  Long Positions: {long_recommendations} ({long_recommendations/total_recommendations:.1%})
  Short Positions: {short_recommendations} ({short_recommendations/total_recommendations:.1%})
  BUY Actions: {buy_actions} ({buy_actions/total_recommendations:.1%})
  SHORT Actions: {short_actions} ({short_actions/total_recommendations:.1%})
  NULL Actions: {null_actions} ({null_actions/total_recommendations:.1%})
  Average P(Win): {avg_p_win:.1%}
  Average Expected Return: {avg_expected_return:.2%}
  Average Kelly Fraction: {avg_kelly:.1%}
  Average EVR Score: {avg_evr_score:.3f}
  Total Risk: ${total_risk:,.0f}
  Available Capital: ${self.risk_guards.current_capital:,.0f}
        """
        
        panel = Panel(summary_text, title="EVR Aggregated Results", border_style="green")
        self.console.print(panel)
    
    def save_aggregated_results(self, aggregated_recommendations: List[Dict[str, Any]], filename_prefix: str = "evr_aggregated") -> None:
        """Save aggregated EVR recommendations to files.
        
        Args:
            aggregated_recommendations: List of aggregated recommendations
            filename_prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Filter out NULL actions to focus on actionable recommendations
        actionable_recommendations = [rec for rec in aggregated_recommendations if rec.get('action') != 'NULL']
        
        # Save as CSV
        csv_file = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
        df = pd.DataFrame(actionable_recommendations)
        df.to_csv(csv_file, index=False)
        self.logger.info(f"Saved {len(actionable_recommendations)} actionable aggregated recommendations to {csv_file}")
        
        # Save as JSON
        json_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(actionable_recommendations, f, indent=2, default=str)
        self.logger.info(f"Saved detailed aggregated results to {json_file}")
        
        # Save summary
        summary_file = self.output_dir / f"{filename_prefix}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"EVR Aggregated Recommendations Summary (Actionable Only)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Recommendations: {len(aggregated_recommendations)} (All)\n")
            f.write(f"Actionable Recommendations: {len(actionable_recommendations)} (BUY/SHORT only)\n")
            f.write(f"Filtered Out: {len(aggregated_recommendations) - len(actionable_recommendations)} (NULL actions)\n\n")
            
            # Calculate statistics for actionable recommendations only
            total_recommendations = len(actionable_recommendations)
            
            if total_recommendations == 0:
                f.write("\nNo actionable recommendations found.\n")
                return
            
            long_recommendations = len([r for r in actionable_recommendations if r['primary_direction'] == 'LONG'])
            short_recommendations = len([r for r in actionable_recommendations if r['primary_direction'] == 'SHORT'])
            buy_actions = len([r for r in actionable_recommendations if r.get('action') == 'BUY'])
            short_actions = len([r for r in actionable_recommendations if r.get('action') == 'SHORT'])
            avg_p_win = sum(r['weighted_p_win'] for r in actionable_recommendations) / total_recommendations
            avg_expected_return = sum(r['weighted_expected_return'] for r in actionable_recommendations) / total_recommendations
            avg_kelly = sum(r['weighted_kelly_fraction'] for r in actionable_recommendations) / total_recommendations
            avg_evr_score = sum(r['evr_composite_score'] for r in actionable_recommendations) / total_recommendations
            total_risk = sum(r['total_risk_dollars'] for r in actionable_recommendations)
            
            f.write(f"Summary Statistics (Actionable Only):\n")
            f.write(f"  Long Positions: {long_recommendations} ({long_recommendations/total_recommendations:.1%})\n")
            f.write(f"  Short Positions: {short_recommendations} ({short_recommendations/total_recommendations:.1%})\n")
            f.write(f"  BUY Actions: {buy_actions} ({buy_actions/total_recommendations:.1%})\n")
            f.write(f"  SHORT Actions: {short_actions} ({short_actions/total_recommendations:.1%})\n")
            f.write(f"  Average P(Win): {avg_p_win:.1%}\n")
            f.write(f"  Average Expected Return: {avg_expected_return:.2%}\n")
            f.write(f"  Average Kelly Fraction: {avg_kelly:.1%}\n")
            f.write(f"  Average EVR Score: {avg_evr_score:.3f}\n")
            f.write(f"  Total Risk: ${total_risk:,.0f}\n")
            f.write(f"  Available Capital: ${self.risk_guards.current_capital:,.0f}\n\n")
            
            f.write(f"Top 15 Actionable Aggregated Recommendations:\n")
            for i, rec in enumerate(actionable_recommendations[:15], 1):
                action = rec.get('action', 'NULL')
                f.write(f"{i:2d}. {rec['ticker']:6s} {rec['primary_direction']:5s} {action:5s} "
                       f"Signals: {rec['total_signals']:2d} P(Win): {rec['weighted_p_win']:5.1%} "
                       f"E[R]: {rec['weighted_expected_return']:6.2%} Kelly: {rec['weighted_kelly_fraction']:5.1%} "
                       f"EVR Score: {rec['evr_composite_score']:.3f}\n")
                f.write(f"     {rec['signal_summary']}\n")
        
        self.logger.info(f"Saved aggregated summary to {summary_file}")
    
    def display_portfolio_status(self) -> None:
        """Display current portfolio status and allocation."""
        summary = self.portfolio_manager.get_portfolio_summary()
        
        # Create portfolio status table
        table = Table(title="Portfolio Status & Allocation")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)
        table.add_column("Details", style="yellow", width=30)
        
        # Capital allocation
        table.add_row(
            "Total Capital",
            f"${summary['total_capital']:,.2f}",
            f"Initial: $1000, P&L: ${summary['total_pnl']:,.2f}"
        )
        
        table.add_row(
            "Available Capital",
            f"${summary['available_capital']:,.2f}",
            f"{(summary['available_capital']/summary['total_capital']*100):.1f}% of total"
        )
        
        table.add_row(
            "Allocated Capital",
            f"${summary['allocated_capital']:,.2f}",
            f"{(summary['allocated_capital']/summary['total_capital']*100):.1f}% of total"
        )
        
        # Performance metrics
        table.add_row(
            "Total Return",
            f"{summary['total_return_pct']:.2%}",
            f"${summary['total_pnl']:,.2f} absolute"
        )
        
        table.add_row(
            "Open Positions",
            str(summary['open_positions']),
            f"Risk: ${summary['allocated_capital']:,.2f}"
        )
        
        table.add_row(
            "Closed Positions",
            str(summary['closed_positions']),
            f"Win Rate: {summary['win_rate']:.1%}"
        )
        
        if summary['closed_positions'] > 0:
            table.add_row(
                "Avg Win",
                f"${summary['avg_win']:,.2f}",
                f"{(summary['avg_win']/summary['total_capital']*100):.1f}%"
            )
            
            table.add_row(
                "Avg Loss",
                f"${summary['avg_loss']:,.2f}",
                f"{(summary['avg_loss']/summary['total_capital']*100):.1f}%"
            )
        
        table.add_row(
            "Run Count",
            str(summary['run_count']),
            f"Last: {summary['last_updated'].strftime('%Y-%m-%d %H:%M')}"
        )
        
        self.console.print(table)
    
    def update_portfolio_from_recommendations(self, recommendations: List[Dict[str, Any]], 
                                             max_positions: int = 5, auto_add: bool = True,
                                             enable_replacement: bool = True,
                                             replacement_threshold: float = 0.20) -> None:
        """Update portfolio with positions from recommendations, with option to replace underperforming positions.
        
        Args:
            recommendations: List of aggregated recommendations
            max_positions: Maximum number of positions to add
            auto_add: Whether to automatically add positions
            enable_replacement: Whether to replace underperforming positions with better opportunities
            replacement_threshold: Minimum EVR score improvement required to replace (0.20 = 20% improvement)
        """
        if not recommendations:
            self.console.print("[yellow]⚠️  No recommendations to process[/yellow]")
            return
        
        # Filter to actionable recommendations (BUY or SHORT action)
        actionable_recs = [r for r in recommendations if r.get('action') in ['BUY', 'SHORT']]
        
        if not actionable_recs:
            self.console.print("[yellow]⚠️  No actionable recommendations (all NULL actions)[/yellow]")
            return
        
        self.console.print(f"\n[cyan]📊 Processing {len(actionable_recs)} actionable recommendations for portfolio...[/cyan]")
        
        # Get current portfolio state
        summary = self.portfolio_manager.get_portfolio_summary()
        available_capital = summary['available_capital']
        open_positions = summary['open_positions']
        
        self.console.print(f"[cyan]Available Capital: ${available_capital:,.2f} | Open Positions: {open_positions}[/cyan]")
        
        # Option 2: Position replacement logic
        if open_positions >= max_positions and enable_replacement:
            self.console.print(f"[yellow]⚠️  Portfolio full ({open_positions}/{max_positions} positions)[/yellow]")
            self.console.print(f"[cyan]🔄 Evaluating position replacement opportunities...[/cyan]")
            
            # Get current open positions with their "EVR scores"
            current_positions = [p for p in self.portfolio_manager.state.positions if p.status == "OPEN"]
            
            # Calculate EVR-like scores for current positions (simplified: p_win * expected_return * kelly)
            position_scores = []
            for pos in current_positions:
                evr_score = pos.p_win * pos.expected_return * pos.kelly_fraction
                days_held = (datetime.now() - pos.entry_date).days
                position_scores.append({
                    'position': pos,
                    'evr_score': evr_score,
                    'days_held': days_held,
                    'ticker': pos.ticker
                })
            
            # Sort by EVR score (ascending - worst first)
            position_scores.sort(key=lambda x: x['evr_score'])
            
            # Get top new recommendations
            top_new_recs = actionable_recs[:3]  # Look at top 3 new opportunities
            
            replaced_count = 0
            for new_rec in top_new_recs:
                if replaced_count >= 2:  # Max 2 replacements per run to avoid excessive churn
                    break
                
                new_evr = new_rec.get('evr_score', 0)
                
                # Find the worst existing position
                if position_scores:
                    worst_position = position_scores[0]
                    
                    # Check if new opportunity is significantly better (20% improvement by default)
                    improvement = (new_evr - worst_position['evr_score']) / worst_position['evr_score'] if worst_position['evr_score'] > 0 else float('inf')
                    
                    if improvement >= replacement_threshold:
                        # Fetch current price to close position
                        try:
                            ticker_data = self.data_fetcher.fetch_multiple_tickers([worst_position['ticker']], period="1d")
                            if worst_position['ticker'] in ticker_data and not ticker_data[worst_position['ticker']].empty:
                                current_price = float(ticker_data[worst_position['ticker']]['Close'].iloc[-1])
                                
                                # Close the underperforming position
                                closed_pos = self.portfolio_manager.close_position(
                                    worst_position['ticker'], 
                                    current_price, 
                                    "REPLACED"
                                )
                                
                                if closed_pos:
                                    replaced_count += 1
                                    pnl_str = f"${closed_pos.pnl:+,.2f}" if closed_pos.pnl else "N/A"
                                    self.console.print(
                                        f"[yellow]🔄 Replaced {worst_position['ticker']} (EVR: {worst_position['evr_score']:.3f}, held {worst_position['days_held']}d) "
                                        f"with {new_rec['ticker']} (EVR: {new_evr:.3f}, +{improvement:.1%} better) - P&L: {pnl_str}[/yellow]"
                                    )
                                    
                                    # Remove from position_scores so we don't replace it again
                                    position_scores.pop(0)
                                    
                                    # Update available capital
                                    available_capital = self.portfolio_manager.state.available_capital
                                    open_positions -= 1
                        except Exception as e:
                            self.logger.error(f"Error replacing position {worst_position['ticker']}: {e}")
            
            if replaced_count == 0:
                self.console.print("[dim]No positions replaced (no significant improvements found)[/dim]")
            else:
                self.console.print(f"[green]✅ Replaced {replaced_count} position(s)[/green]")
            
            # Update summary after replacements
            summary = self.portfolio_manager.get_portfolio_summary()
            available_capital = summary['available_capital']
            open_positions = summary['open_positions']
        
        # If still full after replacement attempts, return
        if open_positions >= max_positions:
            if not enable_replacement:
                self.console.print(f"[yellow]⚠️  Portfolio full ({open_positions}/{max_positions} positions)[/yellow]")
            return
        
        # Add positions up to max_positions limit
        positions_to_add = min(len(actionable_recs), max_positions - open_positions)
        added_count = 0
        skipped_count = 0
        
        for i, rec in enumerate(actionable_recs[:positions_to_add + 5]):  # Look at a few extra in case some fail
            if added_count >= positions_to_add:
                break
            
            ticker = rec['ticker']
            action = rec['action']
            entry_price = rec['entry_price']
            stop_loss = rec['stop_loss']
            
            # Create a TradePlan from the recommendation
            # Calculate avg_r_win and avg_r_loss from expected return
            risk_per_share = abs(entry_price - stop_loss)
            target_price = rec.get('take_profit', entry_price * 1.10)
            reward_per_share = abs(target_price - entry_price) if action == 'BUY' else abs(entry_price - target_price)
            avg_r_win = reward_per_share / risk_per_share if risk_per_share > 0 else 1.0
            avg_r_loss = 1.0  # By definition, loss is 1R
            
            # FIX: Recalculate position size based on ACTUAL available capital
            # Use Kelly fraction but cap risk at 2.5% of actual available capital
            kelly_fraction = rec['weighted_kelly_fraction']
            max_risk_dollars = min(available_capital * 0.025, available_capital * kelly_fraction)
            
            # Calculate shares based on risk per share
            if risk_per_share > 0:
                shares = int(max_risk_dollars / risk_per_share)
                actual_position_value = shares * entry_price
                actual_risk_dollars = shares * risk_per_share
            else:
                shares = 0
                actual_position_value = 0
                actual_risk_dollars = 0
            
            # Skip if position is too small or we don't have enough capital
            if shares <= 0 or actual_position_value > available_capital:
                skipped_count += 1
                self.console.print(f"[yellow]⚠️  Skipped: {ticker} (shares: {shares}, value: ${actual_position_value:.0f} > available ${available_capital:.2f})[/yellow]")
                continue
            
            trade_plan = TradePlan(
                ticker=ticker,
                signal_type=rec['primary_direction'],
                setup=rec.get('best_setup', 'AGGREGATED'),
                entry=entry_price,
                stop=stop_loss,
                take_profit=rec.get('take_profit', entry_price * 1.10),
                targets=[rec.get('take_profit', entry_price * 1.10)],
                position_size=actual_position_value,  # Dollar value of position ✅
                shares=shares,                         # Number of shares ✅
                risk_dollars=actual_risk_dollars,    # Use recalculated risk
                p_win=rec['weighted_p_win'],
                avg_r_win=avg_r_win,
                avg_r_loss=avg_r_loss,
                expected_return=rec['weighted_expected_return'],
                kelly_fraction=rec['weighted_kelly_fraction'],
                confidence=rec.get('best_confidence', 0.5),
                cost_bps=rec.get('avg_cost_bps', 10),
                slippage_bps=rec.get('avg_slippage_bps', 5),
                action=action,
                notes=f"Aggregated from {rec.get('total_signals', 1)} signals - sized for ${available_capital:.0f} capital"
            )
            
            # Try to add position
            if auto_add:
                success = self.portfolio_manager.add_position(trade_plan)
                if success:
                    added_count += 1
                    # Update available capital for next iteration
                    available_capital -= actual_position_value
                    self.console.print(f"[green]✅ Added: {ticker} ({action}) @ ${entry_price:.2f}, Stop: ${stop_loss:.2f}, Shares: {shares}, Risk: ${trade_plan.risk_dollars:,.0f}[/green]")
                else:
                    skipped_count += 1
                    self.console.print(f"[yellow]⚠️  Skipped: {ticker} (insufficient capital)[/yellow]")
            else:
                # Just show what would be added
                self.console.print(f"[cyan]Would add: {ticker} ({action}) @ ${entry_price:.2f}, Risk: ${trade_plan.risk_dollars:,.0f}[/cyan]")
        
        if auto_add:
            self.console.print(f"\n[green]✅ Portfolio updated: {added_count} positions added, {skipped_count} skipped[/green]")
            # Display updated portfolio status
            self.display_portfolio_status()
        else:
            self.console.print(f"\n[cyan]Preview: {positions_to_add} positions would be added[/cyan]")
    
    def backtest_strategy(self, tickers: List[str], start_date: str = "2023-01-01", 
                         end_date: str = "2024-01-01", initial_capital: float = 100000,
                         max_positions: int = 10, rebalance_frequency: str = "weekly") -> Dict[str, Any]:
        """Backtest the EVR scanner strategy.
        
        Args:
            tickers: List of ticker symbols to backtest
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Initial capital for backtest
            max_positions: Maximum number of positions to hold
            rebalance_frequency: How often to rebalance (daily, weekly, monthly)
            
        Returns:
            Dictionary with backtest results and metrics
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        self.logger.info(f"Initial capital: ${initial_capital:,.0f}, Max positions: {max_positions}")
        
        # Initialize portfolio tracking
        portfolio = {
            'capital': initial_capital,
            'positions': {},  # {ticker: {'shares': int, 'entry_price': float, 'entry_date': str}}
            'cash': initial_capital,
            'total_value': initial_capital,
            'trades': [],
            'daily_values': [],
            'dates': [],
            'price_cache': {}  # FIX: Cache for historical prices {(ticker, date): price}
        }
        
        # INTRADAY: Generate datetime range for backtesting (30-minute bars)
        # Parse dates and set to market open if no time specified
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            start_dt = start_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            end_dt = end_dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # INTRADAY: Support 30-minute rebalancing intervals
        if rebalance_frequency == "30min":
            rebalance_minutes = 30
        elif rebalance_frequency == "hourly":
            rebalance_minutes = 60
        elif rebalance_frequency == "daily":
            rebalance_minutes = 390  # 6.5 hours = 1 trading day
        elif rebalance_frequency == "weekly":
            rebalance_minutes = 390 * 5  # 5 trading days
        else:
            rebalance_minutes = 30  # Default to 30-minute for intraday
        
        current_date = start_dt
        last_rebalance_date = start_dt
        
        while current_date <= end_dt:
            # INTRADAY: Skip non-market hours
            if not self._is_market_open(current_date):
                current_date = self._get_next_market_time(current_date, 30)
                continue
            
            # INTRADAY: Check exit conditions every 30 minutes
            self._check_exit_conditions(portfolio, current_date)
            
            # Check if it's time to rebalance
            minutes_since_rebalance = (current_date - last_rebalance_date).total_seconds() / 60
            if minutes_since_rebalance >= rebalance_minutes:
                self.logger.debug(f"Rebalancing on {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # INTRADAY: Get current recommendations for this datetime
                recommendations = self._get_recommendations_for_date(tickers, current_date.strftime('%Y-%m-%d %H:%M:%S'))
                
                self.logger.debug(f"Found {len(recommendations)} recommendations for {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if recommendations:
                    # Execute trades based on recommendations
                    self._execute_rebalance(portfolio, recommendations, current_date, max_positions)
                
                last_rebalance_date = current_date
            
            # INTRADAY: Update portfolio value every 30 minutes
            self._update_portfolio_value(portfolio, current_date)
            
            # INTRADAY: Move to next 30-minute bar
            current_date = self._get_next_market_time(current_date, 30)
        
        # Close all remaining positions at the end of backtest
        self.logger.info("Closing all remaining positions at end of backtest")
        for ticker in list(portfolio['positions'].keys()):
            self._close_position(portfolio, ticker, end_dt)
        
        # INTRADAY: Update final portfolio value after closing all positions
        # At this point, all positions are closed, so total value = cash
        portfolio['total_value'] = portfolio['cash']
        portfolio['daily_values'].append(portfolio['cash'])
        portfolio['dates'].append(end_dt.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Calculate final metrics
        metrics = self._calculate_backtest_metrics(portfolio, start_date, end_date)
        
        # Calculate benchmark metrics
        benchmark_metrics = self._calculate_benchmark_metrics(start_date, end_date, initial_capital)
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(portfolio, benchmark_metrics)
        
        self.logger.info(f"Backtest completed. Final value: ${portfolio['total_value']:,.0f}")
        return {
            'portfolio': portfolio,
            'metrics': metrics,
            'benchmark_metrics': benchmark_metrics,
            'validation_metrics': validation_metrics,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital
        }
    
    def _get_recommendations_for_date(self, tickers: List[str], date: str) -> List[Dict[str, Any]]:
        """Get recommendations for a specific date (simulated historical scanning).
        
        Args:
            tickers: List of ticker symbols
            date: Date to get recommendations for
            
        Returns:
            List of recommendations for that date
        """
        try:
            # Get data up to the specified date
            recommendations = []
            
            # Sample a subset of tickers for backtesting (to keep it manageable)
            sample_tickers = tickers[:min(50, len(tickers))]
            
            for ticker in sample_tickers:
                try:
                    # Get historical data up to the date
                    data = self.get_stock_data(ticker, period="2y", use_cache=True)
                    if data is None or len(data) < 50:
                        self.logger.debug(f"No data for {ticker} (got {len(data) if data is not None else 0} rows)")
                        continue
                    
                    # For backtesting, get historical data up to the specific date
                    # Use yfinance to get data up to the backtest date with retry logic
                    import yfinance as yf
                    
                    # INTRADAY: Calculate start date for 30-minute bars (max 60 days for yfinance)
                    # Use 60 days to get enough 30-min bars for indicators
                    date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S') if ' ' in date else datetime.strptime(date, '%Y-%m-%d')
                    start_date = (date_obj - timedelta(days=60)).strftime('%Y-%m-%d')
                    end_date = date_obj.strftime('%Y-%m-%d %H:%M:%S') if ' ' in date else date
                    
                    # FIX: Get 30-minute intraday data up to the backtest datetime
                    # IMPORTANT: interval="30m" for intraday, end=date ensures point-in-time data
                    # This prevents look-ahead bias by ensuring indicators are calculated with point-in-time data
                    hist_data = None
                    for attempt in range(3):  # Try up to 3 times
                        try:
                            ticker_obj = yf.Ticker(ticker)
                            hist_data = ticker_obj.history(start=start_date, end=end_date, interval="30m")
                            if not hist_data.empty:
                                break
                        except Exception as e:
                            if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                                self.logger.debug(f"Rate limited for {ticker} backtest data, waiting {wait_time}s (attempt {attempt + 1}/3)")
                                time.sleep(wait_time)
                                continue
                            else:
                                self.logger.debug(f"Error fetching backtest data for {ticker}: {e}")
                                break
                    
                    if hist_data is None or hist_data.empty or len(hist_data) < 50:
                        self.logger.debug(f"Not enough historical data for {ticker} up to {date}: {len(hist_data) if hist_data is not None else 0} < 50")
                        continue
                    
                    # Use the historical data
                    data = hist_data
                    
                    self.logger.debug(f"{ticker}: Using {len(data)} rows of historical data up to {date}")
                    
                    if len(data) < 50:
                        self.logger.debug(f"Not enough data for {ticker}: {len(data)} < 50")
                        continue
                    
                    # Calculate indicators
                    indicators = self.calculate_technical_indicators(data)
                    if not indicators:
                        continue
                    
                    # Generate signals
                    signals = self.generate_signals(ticker, data, indicators)
                    self.logger.debug(f"Generated {len(signals)} signals for {ticker} on {date}")
                    if not signals:
                        continue
                    
                    # Create trade plans
                    trade_plans = []
                    for signal in signals:
                        trade_plan = self.create_trade_plan(signal, data)
                        if trade_plan:
                            trade_plans.append(trade_plan)
                    
                    if trade_plans:
                        # Aggregate trade plans for this ticker
                        aggregated = self.aggregate_trade_plans(trade_plans)
                        if aggregated:
                            recommendations.extend(aggregated)
                
                except Exception as e:
                    self.logger.debug(f"Error processing {ticker} for {date}: {e}")
                    time.sleep(0.5)  # Rate limiting for backtest data fetching
                    continue
            
            # Sort by EVR score and return top recommendations
            recommendations.sort(key=lambda x: x['evr_composite_score'], reverse=True)
            self.logger.debug(f"Generated {len(recommendations)} recommendations for {date}")
            return recommendations[:20]  # Top 20 recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations for {date}: {e}")
            return []
    
    def _execute_rebalance(self, portfolio: Dict[str, Any], recommendations: List[Dict[str, Any]], 
                          current_date: datetime, max_positions: int) -> None:
        """Execute portfolio rebalancing based on recommendations.
        
        Args:
            portfolio: Portfolio tracking dictionary
            recommendations: List of recommendations
            current_date: Current date
            max_positions: Maximum number of positions
        """
        try:
            # FIX: Smart rebalancing with cost-benefit analysis
            current_tickers = set(portfolio['positions'].keys())
            recommended_tickers = set(rec['ticker'] for rec in recommendations[:max_positions])
            
            # Evaluate which positions to close based on multiple criteria
            positions_to_close = []
            for ticker in current_tickers:
                if ticker not in recommended_tickers:
                    position = portfolio['positions'][ticker]
                    
                    # Get current price to evaluate P&L
                    current_price = self._get_historical_price(ticker, current_date, portfolio.get('price_cache'))
                    if current_price is None:
                        # Can't get price, close it
                        positions_to_close.append(ticker)
                        continue
                    
                    # Calculate current P&L
                    entry_price = position['entry_price']
                    direction = position['direction']
                    if direction == 'LONG':
                        pnl_pct = (current_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - current_price) / entry_price
                    
                    # Calculate holding period
                    entry_date = datetime.strptime(position['entry_date'], '%Y-%m-%d')
                    holding_days = (current_date - entry_date).days
                    
                    # Decision criteria for closing:
                    # 1. Large loss (>-10%) - cut losses
                    # 2. Been holding for a long time (>30 days) without great performance
                    # 3. Small position (<1% of portfolio) - clean up
                    position_value = abs(position['shares'] * current_price)
                    position_pct = position_value / portfolio['capital']
                    
                    should_close = (
                        pnl_pct < -0.10 or  # Large loss
                        (holding_days > 30 and pnl_pct < 0.05) or  # Stale position
                        position_pct < 0.01  # Very small position
                    )
                    
                    if should_close:
                        positions_to_close.append(ticker)
                    else:
                        # Keep position even though not in top recommendations
                        self.logger.debug(f"Keeping {ticker} despite not in top recs (P&L: {pnl_pct:.1%}, Days: {holding_days})")
            
            # Close selected positions
            for ticker in positions_to_close:
                self._close_position(portfolio, ticker, current_date)
            
            # Open new positions based on recommendations
            for rec in recommendations[:max_positions]:
                ticker = rec['ticker']
                
                if ticker not in portfolio['positions']:
                    # Check if we have room for more positions
                    if len(portfolio['positions']) >= max_positions:
                        break
                    # Open new position
                    self._open_position(portfolio, rec, current_date)
                else:
                    # Position already exists, check if we should adjust
                    self._adjust_position(portfolio, rec, current_date)
        
        except Exception as e:
            self.logger.error(f"Error executing rebalance: {e}")
    
    def _open_position(self, portfolio: Dict[str, Any], recommendation: Dict[str, Any], 
                      current_date: datetime) -> None:
        """Open a new position based on recommendation.
        
        Args:
            portfolio: Portfolio tracking dictionary
            recommendation: Recommendation dictionary
            current_date: Current date
        """
        try:
            ticker = recommendation['ticker']
            entry_price = recommendation['entry_price']
            stop_loss = recommendation['stop_loss']
            direction = recommendation['primary_direction']
            
            # FIX: Proper position sizing with portfolio heat limits
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share <= 0:
                return
            
            # FIX: Check portfolio-level risk limits before opening position
            # Calculate current total risk across all positions
            total_risk = 0
            for pos_ticker, pos in portfolio['positions'].items():
                pos_risk = pos['shares'] * pos.get('risk_per_share', abs(pos['entry_price'] - pos.get('stop_loss', pos['entry_price'] * 0.95)))
                total_risk += pos_risk
            
            # Max total portfolio risk: 20% of capital
            max_total_risk = portfolio['capital'] * 0.20
            if total_risk >= max_total_risk:
                self.logger.debug(f"Portfolio risk limit reached ({total_risk/portfolio['capital']:.1%}), skipping {ticker}")
                return
            
            # Get Kelly fraction but apply conservative limits
            kelly_fraction = recommendation['weighted_kelly_fraction']
            
            # Apply portfolio heat constraints (max 2.5% risk per position)
            max_risk_per_position = portfolio['capital'] * 0.025  # 2.5% max risk
            
            # Calculate position size based on risk
            # Risk = shares * risk_per_share, so shares = max_risk / risk_per_share
            shares_from_risk = int(max_risk_per_position / risk_per_share)
            
            # Also apply Kelly constraint (but capped at 10% of capital)
            kelly_capped = min(kelly_fraction, 0.10)
            shares_from_kelly = int((kelly_capped * portfolio['capital']) / entry_price)
            
            # Take the minimum of the two constraints
            shares = min(shares_from_risk, shares_from_kelly)
            
            if shares <= 0:
                return
            
            actual_cost = shares * entry_price
            
            # Minimum position size ($100)
            if actual_cost < 100:
                return
            
            # Check if we have enough cash
            if actual_cost > portfolio['cash']:
                return
            
            # INTRADAY: Record the trade with datetime
            trade = {
                'date': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                'ticker': ticker,
                'action': 'BUY' if direction == 'LONG' else 'SELL',
                'shares': shares,
                'price': entry_price,
                'value': actual_cost,
                'direction': direction,
                'stop_loss': stop_loss,
                'risk_per_share': risk_per_share
            }
            
            portfolio['trades'].append(trade)
            
            # INTRADAY: Update portfolio - store stop loss and target for exit monitoring
            portfolio['positions'][ticker] = {
                'shares': shares,
                'entry_price': entry_price,
                'entry_date': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                'direction': direction,
                'stop_loss': stop_loss,
                'take_profit': recommendation.get('take_profit', None),
                'risk_per_share': risk_per_share  # Store for risk tracking
            }
            
            # FIX: Proper cash handling for LONG and SHORT positions
            if direction == 'LONG':
                # LONG: pay cash for shares
                portfolio['cash'] -= actual_cost
            else:  # SHORT
                # SHORT: Receive proceeds from short sale, but must post margin
                # Reg T requires 150% of short value (100% proceeds + 50% additional margin)
                margin_required = actual_cost * 1.5
                if margin_required > portfolio['cash']:
                    # Not enough cash for margin, skip this trade
                    portfolio['trades'].pop()  # Remove the trade we just added
                    return
                # Lock up margin requirement (proceeds go to broker, we post additional margin)
                portfolio['cash'] -= margin_required
                # Track the short liability separately
                portfolio['positions'][ticker]['short_liability'] = actual_cost
            
            self.logger.debug(f"Opened {direction} position: {ticker} {shares} shares @ ${entry_price:.2f}, stop: ${stop_loss:.2f}")
        
        except Exception as e:
            self.logger.error(f"Error opening position for {recommendation['ticker']}: {e}")
    
    def _close_position(self, portfolio: Dict[str, Any], ticker: str, current_date: datetime) -> None:
        """Close an existing position.
        
        Args:
            portfolio: Portfolio tracking dictionary
            ticker: Ticker symbol to close
            current_date: Current date
        """
        try:
            if ticker not in portfolio['positions']:
                return
            
            position = portfolio['positions'][ticker]
            shares = position['shares']
            entry_price = position['entry_price']
            direction = position['direction']
            
            # Get actual historical price for the date
            current_price = self._get_historical_price(ticker, current_date, portfolio.get('price_cache'))
            if current_price is None:
                self.logger.warning(f"No price data for {ticker} on {current_date.strftime('%Y-%m-%d')}")
                return
            
            # Calculate P&L
            if direction == 'LONG':
                pnl = (current_price - entry_price) * shares
            else:  # SHORT
                pnl = (entry_price - current_price) * shares
            
            # FIX: Calculate transaction costs using actual ATR from historical data
            trade_value = shares * current_price
            
            # Get actual ATR for this ticker
            atr = self._get_ticker_atr(ticker, current_date)
            if atr is None:
                # Fallback to 2% of price if ATR not available
                atr = current_price * 0.02
            
            commission_cost, slippage_cost = self.cost_model.calculate_costs(
                current_price, atr, trade_value
            )
            total_cost = commission_cost + slippage_cost
            
            # INTRADAY: Record the trade with datetime
            trade = {
                'date': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                'ticker': ticker,
                'action': 'SELL' if direction == 'LONG' else 'BUY',
                'shares': shares,
                'price': current_price,
                'value': trade_value,
                'direction': direction,
                'pnl': pnl,
                'commission': commission_cost,
                'slippage': slippage_cost,
                'net_pnl': pnl - total_cost
            }
            
            portfolio['trades'].append(trade)
            
            # FIX: Proper cash handling for closing LONG and SHORT positions
            if direction == 'LONG':
                # LONG: receive cash from sale, minus costs
                portfolio['cash'] += trade_value - total_cost
            else:  # SHORT
                # SHORT: Buy back shares to cover, pay from cash
                # Return the margin locked up (150% of original short value)
                original_short_value = shares * entry_price
                margin_locked = original_short_value * 1.5
                
                # Cost to buy back shares
                buyback_cost = trade_value + total_cost
                
                # Net cash change: return margin, pay for buyback
                portfolio['cash'] += margin_locked - buyback_cost
            
            del portfolio['positions'][ticker]
            
            self.logger.debug(f"Closed {direction} position: {ticker} {shares} shares @ ${current_price:.2f}, P&L: ${pnl:.2f}, Net: ${pnl - total_cost:.2f}")
        
        except Exception as e:
            self.logger.error(f"Error closing position for {ticker}: {e}")
    
    def _adjust_position(self, portfolio: Dict[str, Any], recommendation: Dict[str, Any], 
                        current_date: datetime) -> None:
        """Adjust an existing position based on new recommendation.
        
        Args:
            portfolio: Portfolio tracking dictionary
            recommendation: New recommendation
            current_date: Current date
        """
        # FIX: Implement position adjustment logic
        # For now, we could update stop-loss to new recommendation's stop
        # But to keep it simple and avoid over-trading, we'll just hold the position
        pass
    
    def _check_exit_conditions(self, portfolio: Dict[str, Any], current_date: datetime) -> None:
        """Check stop-loss and take-profit conditions for all positions.
        
        Args:
            portfolio: Portfolio tracking dictionary
            current_date: Current date
        """
        try:
            tickers_to_close = []
            
            for ticker, position in portfolio['positions'].items():
                # Get current price
                current_price = self._get_historical_price(ticker, current_date, portfolio.get('price_cache'))
                if current_price is None:
                    continue
                
                direction = position['direction']
                entry_price = position['entry_price']
                stop_loss = position.get('stop_loss')
                take_profit = position.get('take_profit')
                
                should_close = False
                reason = ""
                
                if direction == 'LONG':
                    # Check stop-loss for long
                    if stop_loss and current_price <= stop_loss:
                        should_close = True
                        reason = "STOPPED_OUT"
                    # Check take-profit for long
                    elif take_profit and current_price >= take_profit:
                        should_close = True
                        reason = "TARGET_HIT"
                else:  # SHORT
                    # Check stop-loss for short (price goes above stop)
                    if stop_loss and current_price >= stop_loss:
                        should_close = True
                        reason = "STOPPED_OUT"
                    # Check take-profit for short (price goes below target)
                    elif take_profit and current_price <= take_profit:
                        should_close = True
                        reason = "TARGET_HIT"
                
                if should_close:
                    tickers_to_close.append((ticker, reason))
            
            # Close positions that hit exit conditions
            for ticker, reason in tickers_to_close:
                self.logger.debug(f"Exit condition triggered for {ticker}: {reason}")
                self._close_position(portfolio, ticker, current_date)
        
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
    
    def _is_market_open(self, dt: datetime) -> bool:
        """Check if the market is open at the given datetime.
        
        Args:
            dt: Datetime to check
            
        Returns:
            True if market is open, False otherwise
        """
        # Check if it's a weekend
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET (naive datetime, assuming ET)
        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= dt < market_close
    
    def _get_next_market_time(self, dt: datetime, interval_minutes: int = 30) -> datetime:
        """Get the next market time after the given datetime.
        
        Args:
            dt: Current datetime
            interval_minutes: Interval in minutes (default 30)
            
        Returns:
            Next market datetime
        """
        next_dt = dt + timedelta(minutes=interval_minutes)
        
        # If we've moved past market close, jump to next day's open
        market_close = next_dt.replace(hour=16, minute=0, second=0, microsecond=0)
        if next_dt >= market_close:
            # Move to next day at market open
            next_dt = next_dt + timedelta(days=1)
            next_dt = next_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Skip weekends
        while next_dt.weekday() >= 5:
            next_dt = next_dt + timedelta(days=1)
            next_dt = next_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        
        return next_dt
    
    def _get_ticker_atr(self, ticker: str, date: datetime, period: int = 14) -> Optional[float]:
        """Get ATR (Average True Range) for a ticker at a specific date.
        
        Args:
            ticker: Ticker symbol
            date: Date to get ATR for
            period: ATR period (default 14)
            
        Returns:
            ATR value or None if not available
        """
        try:
            import yfinance as yf
            
            # INTRADAY: Get 30-minute data for ATR calculation (need more data points)
            start_date = (date - timedelta(days=period * 2)).strftime('%Y-%m-%d')
            end_date = (date + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
            
            ticker_obj = yf.Ticker(ticker)
            hist_data = ticker_obj.history(start=start_date, end=end_date, interval="30m")
            
            if hist_data.empty or len(hist_data) < period:
                return None
            
            # Calculate True Range
            hist_data['H-L'] = hist_data['High'] - hist_data['Low']
            hist_data['H-PC'] = abs(hist_data['High'] - hist_data['Close'].shift(1))
            hist_data['L-PC'] = abs(hist_data['Low'] - hist_data['Close'].shift(1))
            hist_data['TR'] = hist_data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            
            # Calculate ATR (simple moving average of TR)
            atr = hist_data['TR'].rolling(window=period).mean().iloc[-1]
            
            return float(atr) if not np.isnan(atr) else None
            
        except Exception as e:
            self.logger.debug(f"Error calculating ATR for {ticker}: {e}")
            return None
    
    def _get_historical_price(self, ticker: str, date: datetime, price_cache: Optional[Dict] = None) -> Optional[float]:
        """Get historical price for a ticker on a specific date with retry logic and caching.
        
        Args:
            ticker: Ticker symbol
            date: Date to get price for
            price_cache: Optional price cache dictionary
            
        Returns:
            Price on that date or None if not available
        """
        # FIX: Use cache if available - INTRADAY: include time in cache key
        cache_key = (ticker, date.strftime('%Y-%m-%d %H:%M:%S'))
        if price_cache is not None and cache_key in price_cache:
            return price_cache[cache_key]
        
        import yfinance as yf
        
        for attempt in range(3):  # Try up to 3 times
            try:
                # INTRADAY: Format datetime for yfinance with time component
                start_date = date.strftime('%Y-%m-%d %H:%M:%S')
                end_date = (date + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
                
                # INTRADAY: Get 30-minute bar data for the specific datetime
                ticker_obj = yf.Ticker(ticker)
                hist_data = ticker_obj.history(start=start_date, end=end_date, interval="30m")
                
                if hist_data.empty:
                    # If no data for that specific time, try to get the closest available time
                    # Get a wider range and find the closest datetime
                    wider_start = (date - timedelta(days=7)).strftime('%Y-%m-%d')
                    wider_end = (date + timedelta(days=7)).strftime('%Y-%m-%d')
                    
                    hist_data = ticker_obj.history(start=wider_start, end=wider_end, interval="30m")
                    
                    if hist_data.empty:
                        return None
                    
                    # Find the closest date to our target date
                    hist_data = hist_data[hist_data.index <= date]
                    if hist_data.empty:
                        return None
                
                # Return the closing price for the date
                price = float(hist_data['Close'].iloc[-1])
                
                # Store in cache
                if price_cache is not None:
                    price_cache[cache_key] = price
                
                time.sleep(0.3)  # Rate limiting between successful requests
                return price
            
            except Exception as e:
                if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                    self.logger.debug(f"Rate limited for {ticker} historical price, waiting {wait_time}s (attempt {attempt + 1}/3)")
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.debug(f"Error getting historical price for {ticker} on {date}: {e}")
                    return None
        
        self.logger.debug(f"Failed to get historical price for {ticker} after 3 attempts")
        return None
    
    def _update_portfolio_value(self, portfolio: Dict[str, Any], current_date: datetime) -> None:
        """Update portfolio value for a given date.
        
        Args:
            portfolio: Portfolio tracking dictionary
            current_date: Current date
        """
        try:
            total_value = portfolio['cash']
            
            # FIX: Properly calculate position values including short positions
            for ticker, position in portfolio['positions'].items():
                shares = position['shares']
                direction = position['direction']
                entry_price = position['entry_price']
                
                # Get actual historical price
                current_price = self._get_historical_price(ticker, current_date, portfolio.get('price_cache'))
                if current_price is None:
                    # If no price data, use entry price as fallback
                    current_price = entry_price
                
                if direction == 'LONG':
                    # LONG: position value is simply shares * current price
                    position_value = shares * current_price
                else:  # SHORT
                    # SHORT: We have margin locked (150% of original value)
                    # Current P&L = (entry_price - current_price) * shares
                    # Our equity = margin_locked + P&L - current_liability
                    original_short_value = shares * entry_price
                    current_liability = shares * current_price
                    margin_locked = original_short_value * 1.5
                    
                    # The short position's contribution to portfolio value
                    # is the margin minus the current liability
                    position_value = margin_locked - current_liability
                
                total_value += position_value
            
            portfolio['total_value'] = total_value
            portfolio['daily_values'].append(total_value)
            # INTRADAY: Use datetime format with time component
            portfolio['dates'].append(current_date.strftime('%Y-%m-%d %H:%M:%S'))
        
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    def _calculate_backtest_metrics(self, portfolio: Dict[str, Any], start_date: str, end_date: str) -> Dict[str, float]:
        """Calculate backtest performance metrics.
        
        Args:
            portfolio: Portfolio tracking dictionary
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            initial_capital = portfolio['capital']
            final_value = portfolio['total_value']
            
            # Calculate total return
            total_return = (final_value - initial_capital) / initial_capital
            
            # Calculate annualized return
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end_dt - start_dt).days
            years = days / 365.25
            annualized_return = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0
            
            # Calculate maximum drawdown
            daily_values = portfolio['daily_values']
            if daily_values:
                peak = daily_values[0]
                max_drawdown = 0
                for value in daily_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                max_drawdown = 0
            
            # INTRADAY: Calculate Sharpe ratio (annualized for 30-minute bars)
            # 30-min bars: 13 bars/day * 252 trading days = 3,276 bars/year
            bars_per_year = 13 * 252  # 3,276 for 30-minute bars
            if daily_values and len(daily_values) > 1:
                returns = []
                for i in range(1, len(daily_values)):
                    bar_return = (daily_values[i] - daily_values[i-1]) / daily_values[i-1]
                    returns.append(bar_return)
                
                if returns:
                    avg_return = sum(returns) / len(returns)
                    return_std = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                    # Annualize: multiply by sqrt(bars_per_year)
                    annualized_sharpe = (avg_return * bars_per_year) / (return_std * (bars_per_year ** 0.5)) if return_std > 0 else 0
                    sharpe_ratio = annualized_sharpe
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Calculate win rate (using net P&L after costs)
            trades = portfolio['trades']
            profitable_trades = [t for t in trades if 'net_pnl' in t and t['net_pnl'] > 0]
            win_rate = len(profitable_trades) / len(trades) if trades else 0
            
            # Calculate average trade return (net after costs)
            trade_returns = [t['net_pnl'] for t in trades if 'net_pnl' in t]
            avg_trade_return = sum(trade_returns) / len(trade_returns) if trade_returns else 0
            
            # Calculate total transaction costs
            total_commission = sum(t.get('commission', 0) for t in trades)
            total_slippage = sum(t.get('slippage', 0) for t in trades)
            total_costs = total_commission + total_slippage
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'avg_trade_return': avg_trade_return,
                'total_trades': len(trades),
                'profitable_trades': len(profitable_trades),
                'final_value': final_value,
                'initial_capital': initial_capital,
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'total_costs': total_costs,
                'cost_ratio': total_costs / initial_capital if initial_capital > 0 else 0
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _calculate_benchmark_metrics(self, start_date: str, end_date: str, initial_capital: float) -> Dict[str, float]:
        """Calculate benchmark (SPY) performance for comparison.
        
        Args:
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            
        Returns:
            Dictionary of benchmark metrics
        """
        try:
            # Get SPY data for the period
            spy_data = self.get_stock_data("SPY", period="2y", use_cache=True)
            if spy_data is None:
                return {}
            
            # Filter data for the backtest period
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            spy_data = spy_data[(spy_data.index >= start_date) & (spy_data.index <= end_date)]
            if len(spy_data) < 2:
                return {}
            
            # Calculate benchmark returns
            start_price = float(spy_data['Close'].iloc[0])
            end_price = float(spy_data['Close'].iloc[-1])
            
            benchmark_return = (end_price - start_price) / start_price
            benchmark_value = initial_capital * (1 + benchmark_return)
            
            # Calculate benchmark daily returns for Sharpe ratio
            spy_data['Returns'] = spy_data['Close'].pct_change()
            daily_returns = spy_data['Returns'].dropna()
            
            if len(daily_returns) > 1:
                avg_return = daily_returns.mean()
                return_std = daily_returns.std()
                benchmark_sharpe = avg_return / return_std if return_std > 0 else 0
            else:
                benchmark_sharpe = 0
            
            return {
                'benchmark_return': benchmark_return,
                'benchmark_value': benchmark_value,
                'benchmark_sharpe': benchmark_sharpe,
                'benchmark_start_price': start_price,
                'benchmark_end_price': end_price
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating benchmark metrics: {e}")
            return {}
    
    def _calculate_validation_metrics(self, portfolio: Dict[str, Any], benchmark_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate statistical validation metrics.
        
        Args:
            portfolio: Portfolio tracking dictionary
            benchmark_metrics: Benchmark performance metrics
            
        Returns:
            Dictionary of validation metrics
        """
        try:
            daily_values = portfolio['daily_values']
            if len(daily_values) < 2:
                return {}
            
            # Calculate daily returns
            daily_returns = []
            for i in range(1, len(daily_values)):
                daily_return = (daily_values[i] - daily_values[i-1]) / daily_values[i-1]
                daily_returns.append(daily_return)
            
            if not daily_returns:
                return {}
            
            # Basic statistics
            mean_return = sum(daily_returns) / len(daily_returns)
            return_std = (sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
            
            # Risk metrics
            var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
            var_99 = np.percentile(daily_returns, 1) if len(daily_returns) > 0 else 0
            
            # INTRADAY: Calmar ratio (annualized return / max drawdown)
            # 30-min bars: 13 bars/day * 252 trading days = 3,276 bars/year
            bars_per_year = 13 * 252
            annualized_return = mean_return * bars_per_year
            max_drawdown = 0
            peak = daily_values[0]
            for value in daily_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # INTRADAY: Information ratio (vs benchmark)
            benchmark_return = benchmark_metrics.get('benchmark_return', 0)
            tracking_error = return_std * (bars_per_year ** 0.5)  # Annualize tracking error
            information_ratio = (mean_return * bars_per_year - benchmark_return) / tracking_error if tracking_error > 0 else 0
            
            # INTRADAY: Sortino ratio (downside deviation)
            downside_returns = [r for r in daily_returns if r < 0]
            downside_std = (sum(r ** 2 for r in downside_returns) / len(downside_returns)) ** 0.5 if downside_returns else 0
            annualized_downside_std = downside_std * (bars_per_year ** 0.5)
            sortino_ratio = annualized_return / annualized_downside_std if annualized_downside_std > 0 else 0
            
            return {
                'mean_daily_return': mean_return,
                'return_volatility': return_std,
                'var_95': var_95,
                'var_99': var_99,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'sortino_ratio': sortino_ratio,
                'downside_deviation': downside_std
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating validation metrics: {e}")
            return {}
    
    def display_backtest_results(self, results: Dict[str, Any]) -> None:
        """Display backtest results in a formatted table.
        
        Args:
            results: Backtest results dictionary
        """
        try:
            portfolio = results['portfolio']
            metrics = results['metrics']
            benchmark_metrics = results.get('benchmark_metrics', {})
            validation_metrics = results.get('validation_metrics', {})
            
            # Display performance metrics
            table = Table(title="EVR Scanner Backtest Results")
            table.add_column("Metric", style="cyan", width=20)
            table.add_column("Strategy", style="green", width=15)
            table.add_column("Benchmark (SPY)", style="yellow", width=15)
            table.add_column("Description", style="blue", width=30)
            
            # Strategy vs Benchmark comparison
            strategy_return = metrics['total_return']
            benchmark_return = benchmark_metrics.get('benchmark_return', 0)
            outperformance = strategy_return - benchmark_return
            
            table.add_row("Total Return", f"{strategy_return:.2%}", f"{benchmark_return:.2%}", "Overall return")
            table.add_row("Outperformance", f"{outperformance:+.2%}", "", "Strategy vs Benchmark")
            table.add_row("Final Value", f"${metrics['final_value']:,.0f}", f"${benchmark_metrics.get('benchmark_value', 0):,.0f}", "Portfolio value")
            table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}", f"{benchmark_metrics.get('benchmark_sharpe', 0):.3f}", "Risk-adjusted return")
            table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}", "", "Largest decline")
            table.add_row("Win Rate", f"{metrics['win_rate']:.2%}", "", "Profitable trades %")
            table.add_row("Total Trades", str(metrics['total_trades']), "", "Number of trades")
            table.add_row("Total Costs", f"${metrics.get('total_costs', 0):,.0f}", "", "Transaction costs")
            table.add_row("Cost Ratio", f"{metrics.get('cost_ratio', 0):.2%}", "", "Costs as % of capital")
            
            self.console.print(table)
            
            # Display validation metrics
            if validation_metrics:
                validation_table = Table(title="Statistical Validation Metrics")
                validation_table.add_column("Metric", style="cyan", width=20)
                validation_table.add_column("Value", style="green", width=15)
                validation_table.add_column("Description", style="blue", width=30)
                
                validation_table.add_row("Calmar Ratio", f"{validation_metrics.get('calmar_ratio', 0):.3f}", "Return vs Max Drawdown")
                validation_table.add_row("Information Ratio", f"{validation_metrics.get('information_ratio', 0):.3f}", "Excess return vs Tracking error")
                validation_table.add_row("Sortino Ratio", f"{validation_metrics.get('sortino_ratio', 0):.3f}", "Return vs Downside risk")
                validation_table.add_row("VaR (95%)", f"{validation_metrics.get('var_95', 0):.2%}", "95% Value at Risk")
                validation_table.add_row("VaR (99%)", f"{validation_metrics.get('var_99', 0):.2%}", "99% Value at Risk")
                validation_table.add_row("Volatility", f"{validation_metrics.get('return_volatility', 0):.2%}", "Daily return volatility")
                validation_table.add_row("Downside Dev", f"{validation_metrics.get('downside_deviation', 0):.2%}", "Downside volatility")
                
                self.console.print(validation_table)
            
            # Display summary
            summary_text = f"""
Backtest Summary:
  Period: {results['start_date']} to {results['end_date']}
  Initial Capital: ${metrics['initial_capital']:,.0f}
  Final Value: ${metrics['final_value']:,.0f}
  Total Return: {metrics['total_return']:.2%}
  Annualized Return: {metrics['annualized_return']:.2%}
  Maximum Drawdown: {metrics['max_drawdown']:.2%}
  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
  Win Rate: {metrics['win_rate']:.2%}
  Total Trades: {metrics['total_trades']}
            """
            
            panel = Panel(summary_text, title="EVR Backtest Summary", border_style="green")
            self.console.print(panel)
        
        except Exception as e:
            self.logger.error(f"Error displaying backtest results: {e}")
    
    def save_backtest_results(self, results: Dict[str, Any], filename_prefix: str = "evr_backtest") -> None:
        """Save backtest results to files.
        
        Args:
            results: Backtest results dictionary
            filename_prefix: Prefix for output files
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results as JSON
            json_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Saved backtest results to {json_file}")
            
            # Save metrics summary
            summary_file = self.output_dir / f"{filename_prefix}_{timestamp}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"EVR Scanner Backtest Results\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Period: {results['start_date']} to {results['end_date']}\n\n")
                
                metrics = results['metrics']
                f.write(f"Performance Metrics:\n")
                f.write(f"  Total Return: {metrics['total_return']:.2%}\n")
                f.write(f"  Annualized Return: {metrics['annualized_return']:.2%}\n")
                f.write(f"  Maximum Drawdown: {metrics['max_drawdown']:.2%}\n")
                f.write(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n")
                f.write(f"  Win Rate: {metrics['win_rate']:.2%}\n")
                f.write(f"  Total Trades: {metrics['total_trades']}\n")
                f.write(f"  Profitable Trades: {metrics['profitable_trades']}\n")
                f.write(f"  Average Trade Return: ${metrics['avg_trade_return']:.2f}\n")
                f.write(f"  Initial Capital: ${metrics['initial_capital']:,.0f}\n")
                f.write(f"  Final Value: ${metrics['final_value']:,.0f}\n")
            
            self.logger.info(f"Saved backtest summary to {summary_file}")
        
        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}")
    
    def walk_forward_analysis(self, tickers: List[str], start_date: str = "2023-01-01", 
                             end_date: str = "2024-01-01", initial_capital: float = 100000,
                             training_window: int = 90, testing_window: int = 30) -> Dict[str, Any]:
        """Perform walk-forward analysis for strategy validation.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            initial_capital: Initial capital
            training_window: Training window in days
            testing_window: Testing window in days
            
        Returns:
            Dictionary with walk-forward results
        """
        self.logger.info(f"Starting walk-forward analysis: {training_window}d training, {testing_window}d testing")
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        results = []
        current_date = start_dt
        
        while current_date + timedelta(days=training_window + testing_window) <= end_dt:
            # Define training and testing periods
            train_start = current_date
            train_end = current_date + timedelta(days=training_window)
            test_start = train_end
            test_end = test_start + timedelta(days=testing_window)
            
            self.logger.info(f"Walk-forward period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            
            # Run backtest for this period
            try:
                period_result = self.backtest_strategy(
                    tickers=tickers,
                    start_date=test_start.strftime('%Y-%m-%d'),
                    end_date=test_end.strftime('%Y-%m-%d'),
                    initial_capital=initial_capital,
                    max_positions=5,
                    rebalance_frequency="weekly"
                )
                
                # Add period information
                period_result['period_start'] = test_start.strftime('%Y-%m-%d')
                period_result['period_end'] = test_end.strftime('%Y-%m-%d')
                period_result['training_start'] = train_start.strftime('%Y-%m-%d')
                period_result['training_end'] = train_end.strftime('%Y-%m-%d')
                
                results.append(period_result)
                
            except Exception as e:
                self.logger.error(f"Error in walk-forward period {test_start.strftime('%Y-%m-%d')}: {e}")
            
            # Move to next period
            current_date += timedelta(days=testing_window)
        
        # Calculate aggregate metrics
        if results:
            total_returns = [r['metrics']['total_return'] for r in results]
            sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in results]
            max_drawdowns = [r['metrics']['max_drawdown'] for r in results]
            win_rates = [r['metrics']['win_rate'] for r in results]
            
            aggregate_metrics = {
                'total_periods': len(results),
                'avg_return': sum(total_returns) / len(total_returns),
                'return_std': (sum((r - sum(total_returns)/len(total_returns))**2 for r in total_returns) / len(total_returns))**0.5,
                'avg_sharpe': sum(sharpe_ratios) / len(sharpe_ratios),
                'avg_max_drawdown': sum(max_drawdowns) / len(max_drawdowns),
                'avg_win_rate': sum(win_rates) / len(win_rates),
                'positive_periods': len([r for r in total_returns if r > 0]),
                'consistency': len([r for r in total_returns if r > 0]) / len(total_returns)
            }
        else:
            aggregate_metrics = {}
        
        return {
            'period_results': results,
            'aggregate_metrics': aggregate_metrics,
            'training_window': training_window,
            'testing_window': testing_window,
            'start_date': start_date,
            'end_date': end_date
        }
    
    def display_walk_forward_results(self, results: Dict[str, Any]) -> None:
        """Display walk-forward analysis results.
        
        Args:
            results: Walk-forward results dictionary
        """
        try:
            aggregate_metrics = results['aggregate_metrics']
            
            # Display aggregate metrics
            table = Table(title="Walk-Forward Analysis Results")
            table.add_column("Metric", style="cyan", width=20)
            table.add_column("Value", style="green", width=15)
            table.add_column("Description", style="blue", width=30)
            
            table.add_row("Total Periods", str(aggregate_metrics.get('total_periods', 0)), "Number of test periods")
            table.add_row("Avg Return", f"{aggregate_metrics.get('avg_return', 0):.2%}", "Average period return")
            table.add_row("Return Std", f"{aggregate_metrics.get('return_std', 0):.2%}", "Return volatility")
            table.add_row("Avg Sharpe", f"{aggregate_metrics.get('avg_sharpe', 0):.3f}", "Average Sharpe ratio")
            table.add_row("Avg Max DD", f"{aggregate_metrics.get('avg_max_drawdown', 0):.2%}", "Average max drawdown")
            table.add_row("Avg Win Rate", f"{aggregate_metrics.get('avg_win_rate', 0):.2%}", "Average win rate")
            table.add_row("Positive Periods", str(aggregate_metrics.get('positive_periods', 0)), "Profitable periods")
            table.add_row("Consistency", f"{aggregate_metrics.get('consistency', 0):.2%}", "Success rate")
            
            self.console.print(table)
            
            # Display period-by-period results
            period_table = Table(title="Period-by-Period Results")
            period_table.add_column("Period", style="cyan", width=12)
            period_table.add_column("Return", style="green", width=10)
            period_table.add_column("Sharpe", style="yellow", width=8)
            period_table.add_column("Max DD", style="red", width=8)
            period_table.add_column("Trades", style="blue", width=8)
            
            for i, period_result in enumerate(results['period_results'][:10]):  # Show first 10 periods
                metrics = period_result['metrics']
                period_table.add_row(
                    f"{i+1}",
                    f"{metrics['total_return']:.2%}",
                    f"{metrics['sharpe_ratio']:.3f}",
                    f"{metrics['max_drawdown']:.2%}",
                    str(metrics['total_trades'])
                )
            
            self.console.print(period_table)
        
        except Exception as e:
            self.logger.error(f"Error displaying walk-forward results: {e}")


def main():
    """Main function to run the official scanner."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='EVR Official Ticker Scanner with Full Framework')
    parser.add_argument('--max-tickers', type=int, default=None, help='Maximum number of tickers to scan (default: all tickers)')
    parser.add_argument('--top', type=int, default=20, help='Number of top trade plans to display')
    parser.add_argument('--output-prefix', type=str, default='evr_aggregated', help='Prefix for output files')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching for ticker data')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all cached ticker data before scanning')
    parser.add_argument('--clear-delisted', action='store_true', help='Clear delisted ticker list and retry all tickers')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    parser.add_argument('--use-ml', action='store_true', help='Use ML classifier for probability estimation')
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital for risk management')
    parser.add_argument('--no-aggregate', action='store_true', help='Disable aggregation and show individual trade plans')
    parser.add_argument('--backtest', action='store_true', help='Run backtest instead of live scan')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-01-01', help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--backtest-capital', type=float, default=100000, help='Initial capital for backtest')
    parser.add_argument('--max-positions', type=int, default=10, help='Maximum positions to hold in backtest')
    parser.add_argument('--rebalance-frequency', type=str, default='weekly', choices=['daily', 'weekly', 'monthly'], help='Rebalancing frequency')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward analysis instead of single backtest')
    parser.add_argument('--training-window', type=int, default=90, help='Training window in days for walk-forward analysis')
    parser.add_argument('--testing-window', type=int, default=30, help='Testing window in days for walk-forward analysis')
    
    # Portfolio management options
    parser.add_argument('--max-holding-days', type=int, default=7, help='Maximum days to hold a position before time exit (default: 7)')
    parser.add_argument('--enable-replacement', action='store_true', default=True, help='Enable position replacement when better opportunities appear')
    parser.add_argument('--no-replacement', action='store_false', dest='enable_replacement', help='Disable position replacement')
    parser.add_argument('--replacement-threshold', type=float, default=0.20, help='Minimum EVR improvement to replace position (default: 0.20 = 20%%)')
    
    # Liquidity guardrails arguments
    parser.add_argument('--min-volume', type=int, default=100000, help='Minimum average daily volume for liquidity')
    parser.add_argument('--min-price', type=float, default=1.0, help='Minimum stock price for liquidity')
    parser.add_argument('--max-price', type=float, default=10000.0, help='Maximum stock price for liquidity')
    parser.add_argument('--max-spread', type=float, default=0.05, help='Maximum bid-ask spread as percentage')
    parser.add_argument('--min-daily-volume', type=int, default=50000, help='Minimum daily volume for current day')
    
    args = parser.parse_args()
    
    console = Console()
    
    # Display header
    header_text = """
EVR Official Ticker Scanner with Full Framework

This scanner implements the complete EVR framework with:
1. Official ticker data from NASDAQ FTP
2. Comprehensive technical analysis
3. Empirical Bayes probability estimation
4. Kelly fraction sizing
5. Cost and slippage modeling
6. Risk management and circuit breakers
7. Expected growth ranking
8. Full TradePlan objects with all EVR fields
9. Optional ML classifier for probability estimation
    """
    
    panel = Panel(header_text, title="EVR Official Scanner", border_style="blue")
    console.print(panel)
    
    try:
        # Clear cache if requested
        if args.clear_cache:
            cache_dir = Path("cache")
            if cache_dir.exists():
                console.print(f"[yellow]Clearing cache directory...[/yellow]")
                import shutil
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(exist_ok=True)
                console.print(f"[green]Cache cleared successfully[/green]")
        
        # Clear delisted tickers if requested
        if args.clear_delisted:
            delisted_file = Path("cache/delisted_tickers.json")
            if delisted_file.exists():
                delisted_file.unlink()
                console.print(f"[green]Delisted ticker list cleared[/green]")
        
        # Initialize scanner with EVR framework
        scanner = OfficialTickerScanner(
            log_level=args.log_level,
            use_ml_classifier=args.use_ml,
            initial_capital=args.initial_capital,
            min_avg_volume=args.min_volume,
            min_price=args.min_price,
            max_price=args.max_price,
            max_bid_ask_spread=args.max_spread,
            min_daily_volume=args.min_daily_volume
        )
        
        # Display portfolio status and update allocation
        console.print("\n[blue]Portfolio Status & Allocation Update...[/blue]")
        scanner.display_portfolio_status()
        
        # Get ticker list
        console.print("\n[blue]Getting official ticker lists...[/blue]")
        if args.no_cache:
            tickers = scanner.get_comprehensive_tickers(use_cache=False)
        else:
            tickers = scanner.get_comprehensive_tickers()
        console.print(f"[green]Found {len(tickers)} official tickers[/green]")
        
        if args.backtest:
            if args.walk_forward:
                # Run walk-forward analysis
                console.print("\n[blue]Running EVR walk-forward analysis...[/blue]")
                console.print(f"[blue]Period: {args.start_date} to {args.end_date}[/blue]")
                console.print(f"[blue]Training Window: {args.training_window} days[/blue]")
                console.print(f"[blue]Testing Window: {args.testing_window} days[/blue]")
                console.print(f"[blue]Initial Capital: ${args.backtest_capital:,.0f}[/blue]")
                
                # Use subset of tickers for walk-forward analysis
                wf_tickers = tickers[:min(50, len(tickers))]  # Limit for performance
                
                wf_results = scanner.walk_forward_analysis(
                    tickers=wf_tickers,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    initial_capital=args.backtest_capital,
                    training_window=args.training_window,
                    testing_window=args.testing_window
                )
                
                # Display walk-forward results
                console.print("\n[green]Displaying walk-forward results...[/green]")
                scanner.display_walk_forward_results(wf_results)
                
                # Save walk-forward results
                console.print("\n[blue]Saving walk-forward results...[/blue]")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wf_file = scanner.output_dir / f"{args.output_prefix}_walkforward_{timestamp}.json"
                with open(wf_file, 'w') as f:
                    json.dump(wf_results, f, indent=2, default=str)
                console.print(f"[green]Saved walk-forward results to {wf_file}[/green]")
                
                console.print(f"\n[green]✅ EVR walk-forward analysis completed successfully![/green]")
                aggregate = wf_results['aggregate_metrics']
                console.print(f"[green]Total Periods: {aggregate.get('total_periods', 0)}[/green]")
                console.print(f"[green]Average Return: {aggregate.get('avg_return', 0):.2%}[/green]")
                console.print(f"[green]Consistency: {aggregate.get('consistency', 0):.2%}[/green]")
                console.print(f"[green]Average Sharpe: {aggregate.get('avg_sharpe', 0):.3f}[/green]")
                
            else:
                # Run single backtest
                console.print("\n[blue]Running EVR backtest...[/blue]")
                console.print(f"[blue]Period: {args.start_date} to {args.end_date}[/blue]")
                console.print(f"[blue]Initial Capital: ${args.backtest_capital:,.0f}[/blue]")
                console.print(f"[blue]Max Positions: {args.max_positions}[/blue]")
                console.print(f"[blue]Rebalance Frequency: {args.rebalance_frequency}[/blue]")
                
                # Use subset of tickers for backtesting
                backtest_tickers = tickers[:min(100, len(tickers))]  # Limit for performance
                
                backtest_results = scanner.backtest_strategy(
                    tickers=backtest_tickers,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    initial_capital=args.backtest_capital,
                    max_positions=args.max_positions,
                    rebalance_frequency=args.rebalance_frequency
                )
                
                # Display backtest results
                console.print("\n[green]Displaying backtest results...[/green]")
                scanner.display_backtest_results(backtest_results)
                
                # Save backtest results
                console.print("\n[blue]Saving backtest results...[/blue]")
                scanner.save_backtest_results(backtest_results, filename_prefix=f"{args.output_prefix}_backtest")
                
                console.print(f"\n[green]✅ EVR backtest completed successfully![/green]")
                console.print(f"[green]Final Value: ${backtest_results['metrics']['final_value']:,.0f}[/green]")
                console.print(f"[green]Total Return: {backtest_results['metrics']['total_return']:.2%}[/green]")
                console.print(f"[green]Annualized Return: {backtest_results['metrics']['annualized_return']:.2%}[/green]")
                console.print(f"[green]Max Drawdown: {backtest_results['metrics']['max_drawdown']:.2%}[/green]")
                console.print(f"[green]Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.3f}[/green]")
                console.print(f"[green]Win Rate: {backtest_results['metrics']['win_rate']:.2%}[/green]")
                console.print(f"[green]Total Trades: {backtest_results['metrics']['total_trades']}[/green]")
                
                # Show benchmark comparison
                benchmark_metrics = backtest_results.get('benchmark_metrics', {})
                if benchmark_metrics:
                    benchmark_return = benchmark_metrics.get('benchmark_return', 0)
                    strategy_return = backtest_results['metrics']['total_return']
                    outperformance = strategy_return - benchmark_return
                    console.print(f"[green]Benchmark (SPY) Return: {benchmark_return:.2%}[/green]")
                    console.print(f"[green]Outperformance: {outperformance:+.2%}[/green]")
            
        else:
            # Show cache stats if not clearing cache
            if not args.clear_cache:
                cache_stats = scanner.get_cache_stats()
                if cache_stats.get("total_files", 0) > 0:
                    console.print(f"\n[blue]Cache: {cache_stats['total_files']} files, {cache_stats['total_size_mb']} MB | Delisted: {cache_stats.get('delisted_tickers', 0)}[/blue]")
            
            # Monitor and auto-close positions that hit stops/targets/time
            console.print("\n[cyan]📊 Monitoring Open Positions...[/cyan]")
            monitor_results = scanner.portfolio_manager.monitor_and_close_positions(
                scanner.data_fetcher, 
                max_holding_days=args.max_holding_days
            )
            
            if monitor_results['monitored'] > 0:
                console.print(
                    f"[cyan]Monitored: {monitor_results['monitored']} | "
                    f"Closed: {monitor_results['closed']} | "
                    f"Stopped Out: {monitor_results['stopped_out']} | "
                    f"Targets Hit: {monitor_results['targets_hit']} | "
                    f"Time Exits: {monitor_results['time_exited']} | "
                    f"Errors: {monitor_results['errors']}[/cyan]"
                )
            else:
                console.print("[dim]No open positions to monitor[/dim]")
            
            # Increment run count
            scanner.portfolio_manager.increment_run_count()
            
            # Display portfolio status after monitoring
            console.print("\n[cyan]💼 Current Portfolio Status:[/cyan]")
            scanner.display_portfolio_status()
            
            # Scan for trade plans
            console.print("\n[blue]Scanning for EVR trade plans...[/blue]")
            max_tickers = args.max_tickers if args.max_tickers is not None else len(tickers)
            trade_plans = scanner.scan_tickers(tickers, max_tickers=max_tickers, use_cache=not args.no_cache)
            
            if not args.no_aggregate:
                # Aggregate signals by ticker
                console.print("\n[green]Aggregating signals by ticker...[/green]")
                aggregated_recommendations = scanner.aggregate_trade_plans(trade_plans)
                
                # Display aggregated results
                console.print("\n[green]Displaying EVR aggregated recommendations...[/green]")
                scanner.display_aggregated_results(aggregated_recommendations, top_n=args.top)
                
                # Save aggregated results
                console.print("\n[blue]Saving EVR aggregated recommendations...[/blue]")
                scanner.save_aggregated_results(aggregated_recommendations, filename_prefix=args.output_prefix)
                
                console.print(f"\n[green]✅ EVR aggregation completed successfully![/green]")
                console.print(f"[green]Generated {len(aggregated_recommendations)} aggregated recommendations from {len(trade_plans)} trade plans[/green]")
                
                # Update portfolio with top recommendations
                console.print("\n[cyan]💼 Updating portfolio with recommendations...[/cyan]")
                scanner.update_portfolio_from_recommendations(
                    aggregated_recommendations, 
                    max_positions=5, 
                    auto_add=True,
                    enable_replacement=args.enable_replacement,
                    replacement_threshold=args.replacement_threshold
                )
                
                # Show top aggregated recommendation details
                if aggregated_recommendations:
                    top_rec = aggregated_recommendations[0]
                    action = top_rec.get('action', 'NULL')
                    action_color = "green" if action in ["BUY", "SHORT"] else "yellow"
                    console.print(f"\n[cyan]Top Aggregated Recommendation: {top_rec['ticker']}[/cyan]")
                    console.print(f"[cyan]Signals: {top_rec['total_signals']} | Direction: {top_rec['primary_direction']} | Action: [{action_color}]{action}[/{action_color}][/cyan]")
                    console.print(f"[cyan]P(Win): {top_rec['weighted_p_win']:.1%} | Expected Return: {top_rec['weighted_expected_return']:.2%} | Kelly: {top_rec['weighted_kelly_fraction']:.1%}[/cyan]")
                    console.print(f"[cyan]Entry: ${top_rec['entry_price']:.2f} | Stop: ${top_rec['stop_loss']:.2f} | Target: ${top_rec['take_profit']:.2f}[/cyan]")
                    console.print(f"[cyan]EVR Score: {top_rec['evr_composite_score']:.3f} | Risk: ${top_rec['total_risk_dollars']:,.0f}[/cyan]")
                    console.print(f"[cyan]Signal Summary: {top_rec['signal_summary']}[/cyan]")
                else:
                    console.print(f"\n[red]❌ No EVR aggregated recommendations found![/red]")
            else:
                # Display individual trade plans
                console.print("\n[blue]Displaying EVR trade plans...[/blue]")
                scanner.display_results(trade_plans, top_n=args.top)
                
                # Save individual results
                console.print("\n[blue]Saving EVR trade plans...[/blue]")
                scanner.save_results(trade_plans, filename_prefix=args.output_prefix)
                
                console.print(f"\n[green]✅ EVR scan completed successfully![/green]")
                console.print(f"[green]Generated {len(trade_plans)} ranked trade plans[/green]")
                
                # Show top recommendation details
                if trade_plans:
                    top_plan = trade_plans[0]
                    # LONG: stop < entry (stop below entry, buy to profit from upward move)
                    # SHORT: stop > entry (stop above entry, sell to profit from downward move)
                    direction = "LONG" if top_plan.entry > top_plan.stop else "SHORT"
                    action_color = "green" if top_plan.action in ["BUY", "SHORT"] else "yellow"
                    console.print(f"\n[cyan]Top Trade Plan: {top_plan.ticker}[/cyan]")
                    console.print(f"[cyan]Setup: {top_plan.setup} | Direction: {direction} | Action: [{action_color}]{top_plan.action}[/{action_color}][/cyan]")
                    console.print(f"[cyan]P(Win): {top_plan.p_win:.1%} | Expected Return: {top_plan.expected_return:.2%} | Kelly: {top_plan.kelly_fraction:.1%}[/cyan]")
                    console.print(f"[cyan]Entry: ${top_plan.entry:.2f} | Stop: ${top_plan.stop:.2f} | Target: ${top_plan.targets[0]:.2f}[/cyan]")
                    console.print(f"[cyan]Position Size: ${top_plan.position_size:,.0f} | Risk: ${top_plan.risk_dollars:,.0f}[/cyan]")
        
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

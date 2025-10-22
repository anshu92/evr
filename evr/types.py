"""Core types and dataclasses for EVR."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Bar:
    """OHLCV bar data."""
    
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted: Optional[float] = None
    
    @property
    def returns(self) -> float:
        """Calculate simple returns."""
        return (self.close - self.open) / self.open if self.open > 0 else 0.0
    
    @property
    def log_returns(self) -> float:
        """Calculate log returns."""
        return np.log(self.close / self.open) if self.open > 0 else 0.0
    
    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3.0
    
    @property
    def true_range(self) -> float:
        """Calculate true range (requires previous bar)."""
        # This is a simplified version - full TR needs previous close
        return self.high - self.low


@dataclass(frozen=True)
class Signal:
    """Trading signal."""
    
    symbol: str
    timestamp: pd.Timestamp
    direction: int  # 1 for long, -1 for short, 0 for neutral
    strength: float  # Signal strength (0-1)
    setup: str  # Name of the setup that generated the signal
    features: Dict[str, float]  # Features used in signal generation
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class TradePlan:
    """Trade execution plan."""
    
    signal: Signal
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float  # Dollar amount or shares
    risk_per_trade: float  # Risk as percentage of portfolio
    expected_return: float  # Expected return ratio
    probability: float  # Win probability
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio."""
        if self.stop_loss == self.entry_price:
            return 0.0
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0.0


@dataclass(frozen=True)
class TradeResult:
    """Completed trade result."""
    
    symbol: str
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    returns: float
    duration_days: float
    setup: str
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0
    
    @property
    def is_loser(self) -> bool:
        """Check if trade was unprofitable."""
        return self.pnl < 0


@dataclass(frozen=True)
class Metrics:
    """Performance metrics."""
    
    # Portfolio metrics
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy: float
    
    # Risk metrics
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    max_consecutive_losses: int
    max_consecutive_wins: int
    
    # Additional metrics
    turnover: float
    avg_trade_duration: float
    best_month: float
    worst_month: float
    
    # Metadata
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None


# Type aliases
Symbol = str
Timestamp = pd.Timestamp
Price = float
Volume = float
Returns = float
Risk = float

# Data structures
Bars = pd.DataFrame
Features = pd.DataFrame
Signals = List[Signal]
TradePlans = List[TradePlan]
TradeResults = List[TradeResult]

# Configuration types
ConfigDict = Dict[str, Any]
FeatureDict = Dict[str, float]
MetadataDict = Dict[str, Any]

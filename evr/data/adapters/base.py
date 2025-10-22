"""Base data adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from ...types import Bars


class DataAdapter(ABC):
    """Abstract base class for data adapters."""
    
    def __init__(self, **kwargs):
        """Initialize adapter with configuration."""
        self.config = kwargs
    
    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1d",
        adjust_splits: bool = True,
        adjust_dividends: bool = True,
    ) -> Bars:
        """Get OHLCV bars for a symbol.
        
        Args:
            symbol: Stock symbol/ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Timeframe (1d, 1h, 5m, etc.)
            adjust_splits: Whether to adjust for stock splits
            adjust_dividends: Whether to adjust for dividends
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ValueError: If symbol is invalid
            ConnectionError: If data source is unavailable
        """
        pass
    
    @abstractmethod
    def get_multiple_bars(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1d",
        adjust_splits: bool = True,
        adjust_dividends: bool = True,
    ) -> dict[str, Bars]:
        """Get OHLCV bars for multiple symbols.
        
        Args:
            symbols: List of stock symbols/tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Timeframe (1d, 1h, 5m, etc.)
            adjust_splits: Whether to adjust for stock splits
            adjust_dividends: Whether to adjust for dividends
            
        Returns:
            Dictionary mapping symbols to DataFrames
            
        Raises:
            ValueError: If any symbol is invalid
            ConnectionError: If data source is unavailable
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available.
        
        Returns:
            True if source is available, False otherwise
        """
        pass
    
    def normalize_bars(self, bars: Bars, symbol: str) -> Bars:
        """Normalize bars data format.
        
        Args:
            bars: Raw bars data
            symbol: Stock symbol
            
        Returns:
            Normalized bars data
        """
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in bars.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Add symbol column
        bars = bars.copy()
        bars['Symbol'] = symbol
        
        # Ensure timestamp index
        if not isinstance(bars.index, pd.DatetimeIndex):
            bars.index = pd.to_datetime(bars.index)
        
        # Sort by timestamp
        bars = bars.sort_index()
        
        # Remove duplicates
        bars = bars[~bars.index.duplicated(keep='last')]
        
        # Ensure numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            bars[col] = pd.to_numeric(bars[col], errors='coerce')
        
        # Validate OHLC relationships
        invalid_mask = (
            (bars['High'] < bars['Low']) |
            (bars['High'] < bars['Open']) |
            (bars['High'] < bars['Close']) |
            (bars['Low'] > bars['Open']) |
            (bars['Low'] > bars['Close'])
        )
        
        if invalid_mask.any():
            bars = bars[~invalid_mask]
        
        return bars

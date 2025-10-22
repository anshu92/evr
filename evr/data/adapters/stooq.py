"""Stooq data adapter."""

from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
import pandas_datareader as pdr

from .base import DataAdapter
from ...types import Bars


class StooqAdapter(DataAdapter):
    """Stooq data adapter for Stooq.com data."""
    
    def __init__(self, **kwargs):
        """Initialize Stooq adapter."""
        super().__init__(**kwargs)
        self.rate_limit_delay = kwargs.get('rate_limit_delay', 0.1)
    
    def get_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1d",
        adjust_splits: bool = True,
        adjust_dividends: bool = True,
    ) -> Bars:
        """Get OHLCV bars from Stooq.
        
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
            ConnectionError: If Stooq is unavailable
        """
        try:
            # Stooq uses different symbol format for US stocks
            stooq_symbol = self._convert_symbol(symbol)
            
            # Download data
            bars = pdr.get_data_stooq(
                stooq_symbol,
                start=start_date,
                end=end_date,
            )
            
            if bars.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # Stooq data is already adjusted
            # Normalize data
            bars = self.normalize_bars(bars, symbol)
            
            return bars
            
        except Exception as e:
            if "No data found" in str(e):
                raise ValueError(f"Symbol not found: {symbol}")
            else:
                raise ConnectionError(f"Failed to fetch data for {symbol}: {e}")
    
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
        """
        results = {}
        
        for symbol in symbols:
            try:
                bars = self.get_bars(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    adjust_splits=adjust_splits,
                    adjust_dividends=adjust_dividends,
                )
                results[symbol] = bars
                
                # Rate limiting
                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                print(f"Warning: Failed to fetch data for {symbol}: {e}")
                continue
        
        return results
    
    def is_available(self) -> bool:
        """Check if Stooq is available."""
        try:
            # Test with a simple request
            test_data = pdr.get_data_stooq("AAPL.US", period="1d")
            return not test_data.empty
        except Exception:
            return False
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbol to Stooq format.
        
        Args:
            symbol: Original symbol
            
        Returns:
            Stooq-formatted symbol
        """
        # Add .US suffix for US stocks
        if not symbol.endswith('.US'):
            return f"{symbol}.US"
        return symbol

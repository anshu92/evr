"""YFinance data adapter."""

from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
import yfinance as yf

from .base import DataAdapter
from ...types import Bars


class YFinanceAdapter(DataAdapter):
    """YFinance data adapter for Yahoo Finance data."""
    
    def __init__(self, **kwargs):
        """Initialize YFinance adapter."""
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
        """Get OHLCV bars from Yahoo Finance.
        
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
            ConnectionError: If Yahoo Finance is unavailable
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Map timeframe to yfinance interval
            interval_map = {
                "1m": "1m",
                "2m": "2m", 
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "60m": "60m",
                "90m": "90m",
                "1h(60m)": "1h",
                "1d": "1d",
                "5d": "5d",
                "1wk": "1wk",
                "1mo": "1mo",
                "3mo": "3mo",
            }
            
            interval = interval_map.get(timeframe, "1d")
            
            # Download data
            bars = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=adjust_splits and adjust_dividends,
                prepost=False,
                threads=True,
            )
            
            if bars.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # Handle adjustments manually if needed
            if not (adjust_splits and adjust_dividends):
                if adjust_splits:
                    bars = self._adjust_splits(bars, ticker)
                if adjust_dividends:
                    bars = self._adjust_dividends(bars, ticker)
            
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
        """Check if Yahoo Finance is available."""
        try:
            # Test with a simple request
            ticker = yf.Ticker("AAPL")
            test_data = ticker.history(period="1d")
            return not test_data.empty
        except Exception:
            return False
    
    def _adjust_splits(self, bars: Bars, ticker: yf.Ticker) -> Bars:
        """Adjust bars for stock splits."""
        try:
            splits = ticker.splits
            if splits.empty:
                return bars
            
            # Apply split adjustments
            for date, split_ratio in splits.items():
                if date in bars.index:
                    bars.loc[date:, ['Open', 'High', 'Low', 'Close']] *= split_ratio
            
            return bars
        except Exception:
            return bars
    
    def _adjust_dividends(self, bars: Bars, ticker: yf.Ticker) -> Bars:
        """Adjust bars for dividends."""
        try:
            dividends = ticker.dividends
            if dividends.empty:
                return bars
            
            # Apply dividend adjustments
            for date, dividend in dividends.items():
                if date in bars.index:
                    adj_factor = 1 - dividend / bars.loc[date, 'Close']
                    bars.loc[date:, ['Open', 'High', 'Low', 'Close']] *= adj_factor
            
            return bars
        except Exception:
            return bars

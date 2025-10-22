"""Parquet-based data cache."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .base import DataCache
from ...types import Bars


class ParquetCache(DataCache):
    """Parquet-based data cache implementation."""
    
    def get(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1d",
    ) -> Optional[Bars]:
        """Get cached data from Parquet file.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            
        Returns:
            Cached data or None if not found/expired
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date, timeframe)
        
        if not cache_path.exists() or self._is_expired(cache_path):
            return None
        
        try:
            # Read Parquet file
            table = pq.read_table(cache_path)
            data = table.to_pandas()
            
            # Ensure timestamp index
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            return data
            
        except Exception as e:
            print(f"Warning: Failed to read cache file {cache_path}: {e}")
            return None
    
    def put(
        self,
        symbol: str,
        data: Bars,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1d",
    ) -> None:
        """Store data in Parquet file.
        
        Args:
            symbol: Stock symbol
            data: Data to cache
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date, timeframe)
        
        try:
            # Prepare data for Parquet
            data_to_save = data.copy()
            
            # Reset index to make timestamp a column
            if isinstance(data_to_save.index, pd.DatetimeIndex):
                data_to_save = data_to_save.reset_index()
                data_to_save.rename(columns={'index': 'timestamp'}, inplace=True)
            
            # Convert to PyArrow table
            table = pa.Table.from_pandas(data_to_save)
            
            # Write to Parquet
            pq.write_table(table, cache_path, compression='snappy')
            
        except Exception as e:
            print(f"Warning: Failed to write cache file {cache_path}: {e}")
    
    def exists(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1d",
    ) -> bool:
        """Check if cached data exists and is valid.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            
        Returns:
            True if valid cached data exists
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date, timeframe)
        return cache_path.exists() and not self._is_expired(cache_path)
    
    def invalidate(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1d",
    ) -> None:
        """Invalidate cached data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date, timeframe)
        
        if cache_path.exists():
            cache_path.unlink()
    
    def get_cache_info(self) -> dict:
        """Get cache information.
        
        Returns:
            Dictionary with cache statistics
        """
        info = super().get_cache_info()
        
        if not self.cache_dir.exists():
            return info
        
        # Add Parquet-specific info
        parquet_files = list(self.cache_dir.glob("*.parquet"))
        info["parquet_files"] = len(parquet_files)
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in parquet_files)
        info["total_size_mb"] = total_size / (1024 * 1024)
        
        return info

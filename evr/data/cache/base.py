"""Base cache interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from ...types import Bars


class DataCache(ABC):
    """Abstract base class for data caches."""
    
    def __init__(self, cache_dir: str = "~/.evr/data", ttl_days: int = 1):
        """Initialize cache.
        
        Args:
            cache_dir: Directory to store cached data
            ttl_days: Time-to-live in days
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days
    
    @abstractmethod
    def get(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1d",
    ) -> Optional[Bars]:
        """Get cached data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            
        Returns:
            Cached data or None if not found/expired
        """
        pass
    
    @abstractmethod
    def put(
        self,
        symbol: str,
        data: Bars,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1d",
    ) -> None:
        """Store data in cache.
        
        Args:
            symbol: Stock symbol
            data: Data to cache
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    def _get_cache_path(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1d",
    ) -> Path:
        """Get cache file path.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            
        Returns:
            Path to cache file
        """
        # Create filename from parameters
        filename_parts = [symbol, timeframe]
        if start_date:
            filename_parts.append(start_date)
        if end_date:
            filename_parts.append(end_date)
        
        filename = "_".join(filename_parts) + ".parquet"
        return self.cache_dir / filename
    
    def _is_expired(self, cache_path: Path) -> bool:
        """Check if cache file is expired.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            True if expired
        """
        if not cache_path.exists():
            return True
        
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = file_time + timedelta(days=self.ttl_days)
        
        return datetime.now() > expiry_time
    
    def clear(self) -> None:
        """Clear all cached data."""
        if self.cache_dir.exists():
            for file_path in self.cache_dir.glob("*.parquet"):
                file_path.unlink()
    
    def get_cache_info(self) -> dict:
        """Get cache information.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir.exists():
            return {"total_files": 0, "total_size": 0, "expired_files": 0}
        
        total_files = 0
        total_size = 0
        expired_files = 0
        
        for file_path in self.cache_dir.glob("*.parquet"):
            total_files += 1
            total_size += file_path.stat().st_size
            
            if self._is_expired(file_path):
                expired_files += 1
        
        return {
            "total_files": total_files,
            "total_size": total_size,
            "expired_files": expired_files,
            "cache_dir": str(self.cache_dir),
        }

"""Universe loader for stock symbols."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


class UniverseLoader:
    """Load and manage stock universe."""
    
    def __init__(self, universe_file: Optional[str] = None):
        """Initialize universe loader.
        
        Args:
            universe_file: Path to universe CSV file
        """
        self.universe_file = universe_file
        self._universe: Optional[pd.DataFrame] = None
    
    def load_universe(self) -> pd.DataFrame:
        """Load universe from file.
        
        Returns:
            DataFrame with universe data
            
        Raises:
            FileNotFoundError: If universe file doesn't exist
            ValueError: If universe file format is invalid
        """
        if self.universe_file is None:
            # Load default S&P 500 universe
            return self._load_default_universe()
        
        universe_path = Path(self.universe_file)
        if not universe_path.exists():
            raise FileNotFoundError(f"Universe file not found: {universe_path}")
        
        try:
            # Load CSV file
            df = pd.read_csv(universe_path)
            
            # Validate required columns
            required_cols = ['symbol', 'name']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Universe file must contain columns: {required_cols}")
            
            # Clean data
            df = df.dropna(subset=['symbol'])
            df['symbol'] = df['symbol'].str.upper().str.strip()
            
            self._universe = df
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load universe file: {e}")
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols from universe.
        
        Returns:
            List of stock symbols
        """
        if self._universe is None:
            self.load_universe()
        
        return self._universe['symbol'].tolist()
    
    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get information for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with symbol information or None if not found
        """
        if self._universe is None:
            self.load_universe()
        
        symbol = symbol.upper().strip()
        row = self._universe[self._universe['symbol'] == symbol]
        
        if row.empty:
            return None
        
        return row.iloc[0].to_dict()
    
    def filter_by_sector(self, sector: str) -> List[str]:
        """Filter symbols by sector.
        
        Args:
            sector: Sector name
            
        Returns:
            List of symbols in the sector
        """
        if self._universe is None:
            self.load_universe()
        
        if 'sector' not in self._universe.columns:
            return []
        
        sector_symbols = self._universe[
            self._universe['sector'].str.contains(sector, case=False, na=False)
        ]
        
        return sector_symbols['symbol'].tolist()
    
    def filter_by_market_cap(self, min_cap: Optional[float] = None, 
                           max_cap: Optional[float] = None) -> List[str]:
        """Filter symbols by market cap.
        
        Args:
            min_cap: Minimum market cap
            max_cap: Maximum market cap
            
        Returns:
            List of symbols matching market cap criteria
        """
        if self._universe is None:
            self.load_universe()
        
        if 'market_cap' not in self._universe.columns:
            return []
        
        filtered = self._universe.copy()
        
        if min_cap is not None:
            filtered = filtered[filtered['market_cap'] >= min_cap]
        
        if max_cap is not None:
            filtered = filtered[filtered['market_cap'] <= max_cap]
        
        return filtered['symbol'].tolist()
    
    def _load_default_universe(self) -> pd.DataFrame:
        """Load default S&P 500 universe.
        
        Returns:
            DataFrame with S&P 500 symbols
        """
        # Default S&P 500 symbols (as of 2025 Q3)
        sp500_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
            "UNH", "JNJ", "V", "PG", "JPM", "XOM", "HD", "CVX", "MA", "PFE",
            "ABBV", "BAC", "AVGO", "PEP", "TMO", "COST", "WMT", "DHR", "ABT",
            "VZ", "ACN", "NFLX", "ADBE", "CRM", "TXN", "NKE", "CMCSA", "QCOM",
            "PM", "NEE", "UNP", "RTX", "AMD", "HON", "SPGI", "AMGN", "T",
            "INTU", "IBM", "GS", "CAT", "AXP", "BKNG", "CVS", "GILD", "LOW",
            "MDT", "PYPL", "SBUX", "TGT", "USB", "VRTX", "WBA", "ZTS", "AON",
            "ADI", "AMT", "CB", "CHTR", "CL", "COF", "COP", "CSX", "DE",
            "DUK", "EMR", "EQIX", "EW", "FIS", "FISV", "GE", "GM", "GOOG",
            "HCA", "ICE", "ISRG", "JCI", "KMB", "KO", "LMT", "MCD", "MMM",
            "MO", "MRK", "MSI", "NOC", "NOW", "ORCL", "PGR", "PLD", "PRU",
            "PSA", "REGN", "ROP", "SYK", "TEL", "TFC", "TJX", "TMO", "TRV",
            "TXN", "UPS", "VLO", "WEC", "WY", "XEL", "YUM", "ZBH"
        ]
        
        # Create DataFrame
        df = pd.DataFrame({
            'symbol': sp500_symbols,
            'name': [f"{symbol} Corp" for symbol in sp500_symbols],
            'sector': ['Technology'] * len(sp500_symbols),  # Simplified
            'market_cap': [1000000] * len(sp500_symbols),  # Placeholder
        })
        
        self._universe = df
        return df

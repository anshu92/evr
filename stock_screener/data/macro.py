"""Fetch macro-economic indicators for regime detection."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf


def fetch_macro_indicators(lookback_days: int = 730, logger=None) -> pd.DataFrame:
    """
    Fetch macro indicators: VIX, 10Y Treasury yield, 2Y Treasury yield.
    Returns daily DataFrame with date index.
    """
    now = datetime.now(tz=timezone.utc)
    start = (now - timedelta(days=int(lookback_days * 1.5))).date().isoformat()
    
    indicators = {
        "^VIX": "vix",  # CBOE Volatility Index
        "^TNX": "treasury_10y",  # 10-Year Treasury Yield
        "^IRX": "treasury_13w",  # 13-Week Treasury Bill
    }
    
    results = {}
    for ticker, name in indicators.items():
        try:
            data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if not data.empty and "Close" in data.columns:
                results[name] = data["Close"]
        except Exception as e:
            if logger:
                logger.warning(f"Failed to fetch {name} ({ticker}): {e}")
            results[name] = pd.Series(dtype=float)
    
    # Combine into single DataFrame
    df = pd.DataFrame(results)
    df.index.name = "date"
    
    # Add derived features
    if "treasury_10y" in df.columns and "treasury_13w" in df.columns:
        # Yield curve slope (10Y - 3M)
        df["yield_curve_slope"] = df["treasury_10y"] - df["treasury_13w"]
    
    # Forward-fill missing values (holidays, etc.)
    df = df.ffill().bfill()
    
    if logger:
        logger.info(
            f"Fetched macro indicators: {len(df)} days, "
            f"columns={list(df.columns)}, date_range={df.index.min()} to {df.index.max()}"
        )
    
    return df

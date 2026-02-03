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
            if data is not None and not data.empty:
                # Handle both single and multi-level column formats from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-level columns: ('Close', 'TICKER')
                    if "Close" in data.columns.get_level_values(0):
                        close_data = data["Close"]
                        if isinstance(close_data, pd.DataFrame):
                            close_data = close_data.iloc[:, 0]
                        results[name] = close_data.squeeze()
                elif "Close" in data.columns:
                    results[name] = data["Close"].squeeze()
        except Exception as e:
            if logger:
                logger.warning(f"Failed to fetch {name} ({ticker}): {e}")
    
    # Combine into single DataFrame
    # Handle case where no data was fetched
    if not results:
        if logger:
            logger.warning("All macro indicator fetches failed, returning empty DataFrame")
        return pd.DataFrame()
    
    # Build DataFrame from successfully fetched series
    df = pd.concat(results, axis=1)
    df.index.name = "date"
    
    # Add derived features
    if not df.empty and "treasury_10y" in df.columns and "treasury_13w" in df.columns:
        # Yield curve slope (10Y - 3M)
        df["yield_curve_slope"] = df["treasury_10y"] - df["treasury_13w"]
    
    # Forward-fill missing values (holidays, etc.)
    if not df.empty:
        df = df.ffill().bfill()
    
    if logger:
        if not df.empty:
            logger.info(
                f"Fetched macro indicators: {len(df)} days, "
                f"columns={list(df.columns)}, date_range={df.index.min()} to {df.index.max()}"
            )
        else:
            logger.info("No macro indicators fetched - proceeding without macro features")
    
    return df

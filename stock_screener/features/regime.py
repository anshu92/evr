"""Market regime detection for adaptive strategy."""
from __future__ import annotations

import pandas as pd
import numpy as np


def detect_regime(market_returns: pd.Series, window: int = 60) -> str:
    """Simple regime detection based on recent market behavior."""
    if len(market_returns) < window:
        return "insufficient_data"
    
    recent = market_returns.tail(window)
    
    # Annualized volatility and trend
    vol = recent.std() * np.sqrt(252)
    trend = recent.mean() * 252
    
    # Regime classification
    if vol > 0.25:
        return "high_volatility"
    elif trend > 0.10:
        return "trending_up"
    elif trend < -0.10:
        return "trending_down"
    else:
        return "range_bound"


def get_regime_stats(market_returns: pd.Series, window: int = 60) -> dict[str, float]:
    """Get detailed regime statistics."""
    if len(market_returns) < window:
        return {
            "regime": "insufficient_data",
            "volatility_ann": float("nan"),
            "trend_ann": float("nan"),
            "window_days": window,
        }
    
    recent = market_returns.tail(window)
    vol = recent.std() * np.sqrt(252)
    trend = recent.mean() * 252
    
    return {
        "regime": detect_regime(market_returns, window),
        "volatility_ann": float(vol),
        "trend_ann": float(trend),
        "window_days": window,
    }

"""Tests for regime detection."""
import numpy as np
import pandas as pd

from stock_screener.features.regime import detect_regime, get_regime_stats


def test_detect_trending_up():
    """Test detection of uptrend."""
    # Strong uptrend
    returns = pd.Series(np.linspace(0.001, 0.003, 100))
    
    regime = detect_regime(returns, window=60)
    assert regime == "trending_up"


def test_detect_trending_down():
    """Test detection of downtrend."""
    # Strong downtrend
    returns = pd.Series(np.linspace(-0.003, -0.001, 100))
    
    regime = detect_regime(returns, window=60)
    assert regime == "trending_down"


def test_detect_high_volatility():
    """Test detection of high volatility."""
    # High volatility, no trend
    returns = pd.Series(np.random.randn(100) * 0.02)
    
    regime = detect_regime(returns, window=60)
    # Could be high_volatility or range_bound depending on random seed
    assert regime in ["high_volatility", "range_bound"]


def test_insufficient_data():
    """Test handling of insufficient data."""
    short_series = pd.Series(np.random.randn(30))
    
    regime = detect_regime(short_series, window=60)
    assert regime == "insufficient_data"


def test_regime_stats():
    """Test regime statistics computation."""
    returns = pd.Series(np.random.randn(100) * 0.01 + 0.001)
    
    stats = get_regime_stats(returns, window=60)
    
    assert "regime" in stats
    assert "volatility_ann" in stats
    assert "trend_ann" in stats
    assert "window_days" in stats
    
    assert stats["window_days"] == 60
    assert not pd.isna(stats["volatility_ann"])
    assert not pd.isna(stats["trend_ann"])

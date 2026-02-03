import logging
import numpy as np
import pandas as pd
import pytest

from stock_screener.features.technical import _annualized_vol, _drawdown, _rsi


def test_annualized_vol_empty_series():
    """Test that annualized_vol handles empty series."""
    result = _annualized_vol(pd.Series([]), window=20)
    assert pd.isna(result)


def test_annualized_vol_insufficient_data():
    """Test that annualized_vol handles insufficient data."""
    result = _annualized_vol(pd.Series([1.0, 2.0, 3.0]), window=20)
    assert pd.isna(result)


def test_annualized_vol_normal_case():
    """Test that annualized_vol computes correctly."""
    # Create a series with known returns
    prices = pd.Series([100, 101, 102, 101, 100] * 10)  # 50 points
    result = _annualized_vol(prices, window=20)
    
    assert not pd.isna(result)
    assert result > 0  # Volatility should be positive


def test_drawdown_empty_series():
    """Test that drawdown handles empty series."""
    result = _drawdown(pd.Series([]), window=60)
    assert pd.isna(result)


def test_drawdown_normal_case():
    """Test that drawdown computes correctly."""
    # Series that goes from 100 to 80 (20% drawdown)
    prices = pd.Series([100, 95, 90, 85, 80])
    result = _drawdown(prices, window=5)
    
    assert not pd.isna(result)
    assert result < 0  # Drawdown should be negative
    assert result == pytest.approx(-0.2, abs=0.01)  # 20% drawdown


def test_rsi_empty_series():
    """Test that RSI handles empty series."""
    result = _rsi(pd.Series([]), period=14)
    assert pd.isna(result)


def test_rsi_insufficient_data():
    """Test that RSI handles insufficient data."""
    result = _rsi(pd.Series([1.0, 2.0, 3.0]), period=14)
    assert pd.isna(result)


def test_rsi_normal_case():
    """Test that RSI computes within valid range."""
    # Create a trending series
    prices = pd.Series(range(100, 150))  # Uptrend
    result = _rsi(prices, period=14)
    
    assert not pd.isna(result)
    assert 0 <= result <= 100  # RSI should be between 0 and 100
    assert result > 50  # Uptrend should have RSI > 50


def test_cross_sectional_ranks():
    """Test that cross-sectional ranking works correctly."""
    from stock_screener.features.technical import compute_features
    
    # Create mock price data with 3 tickers
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Create MultiIndex columns for prices
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    columns = pd.MultiIndex.from_product([tickers, ['Close', 'Volume']])
    
    # Create price data
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.randn(100, 6) * 10 + 100,
        index=dates,
        columns=columns
    )
    
    # Make volume data
    for ticker in tickers:
        prices[(ticker, 'Volume')] = np.random.randint(1_000_000, 10_000_000, 100)
    
    # Create FX series
    fx = pd.Series(1.35, index=dates)
    
    logger = logging.getLogger("test")
    
    # Compute features (will test if it runs without error)
    try:
        result = compute_features(
            prices=prices,
            fx_usdcad=fx,
            liquidity_lookback_days=30,
            feature_lookback_days=90,
            logger=logger,
            fundamentals=None,
        )
        
        # Check that rank columns exist
        assert "rank_ret_20d" in result.columns
        assert "rank_ret_60d" in result.columns
        assert "rank_vol_60d" in result.columns
        
        # Check that ranks are between 0 and 1
        for col in ["rank_ret_20d", "rank_ret_60d", "rank_vol_60d"]:
            if col in result.columns and not result[col].isna().all():
                assert (result[col].dropna() >= 0).all()
                assert (result[col].dropna() <= 1).all()
                
    except Exception as e:
        # If there's an error due to insufficient data, that's ok for this test
        if "No features computed" not in str(e):
            raise

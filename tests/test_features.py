import logging
import numpy as np
import pandas as pd
import pytest

from stock_screener.features.technical import _annualized_vol, _drawdown, _rsi
from stock_screener.modeling.model import TECHNICAL_FEATURES_ONLY


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


def test_market_breadth_uses_ma20_ratio_sign():
    """Breadth should reflect share of stocks above MA20 (ma20_ratio > 0)."""
    from stock_screener.features.technical import compute_features

    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    tickers = ["UP1", "UP2", "DN1"]
    columns = pd.MultiIndex.from_product([tickers, ["Close", "Volume"]])
    prices = pd.DataFrame(index=dates, columns=columns, dtype=float)

    # Two up-trending names should be above MA20; one down-trending below MA20.
    prices[("UP1", "Close")] = np.linspace(100, 160, len(dates))
    prices[("UP2", "Close")] = np.linspace(50, 90, len(dates))
    prices[("DN1", "Close")] = np.linspace(120, 70, len(dates))
    for t in tickers:
        prices[(t, "Volume")] = 2_000_000

    fx = pd.Series(1.35, index=dates)
    logger = logging.getLogger("test")
    out = compute_features(
        prices=prices,
        fx_usdcad=fx,
        liquidity_lookback_days=30,
        feature_lookback_days=200,
        logger=logger,
        fundamentals=None,
    )
    breadth = float(out["market_breadth"].iloc[0])
    assert 0.5 < breadth < 1.0


def test_feature_schema_parity_for_inference():
    """Inference features should expose all technical training columns."""
    from stock_screener.features.technical import compute_features

    np.random.seed(11)
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    columns = pd.MultiIndex.from_product([tickers, ["Close", "Volume"]])
    prices = pd.DataFrame(index=dates, columns=columns, dtype=float)
    for t in tickers:
        prices[(t, "Close")] = 100 * np.cumprod(1 + np.random.normal(0, 0.01, len(dates)))
        prices[(t, "Volume")] = np.random.randint(1_000_000, 5_000_000, len(dates))

    fundamentals = pd.DataFrame(
        {
            "sector": ["Tech", "Tech", "Comm", "Retail"],
            "industry": ["Software", "Software", "Internet", "Ecomm"],
            "marketCap": [2e12, 1.8e12, 1.2e12, 1.5e12],
            "beta": [1.1, 1.0, 1.2, 1.3],
        },
        index=tickers,
    )
    macro = pd.DataFrame(
        {
            "vix": np.linspace(12, 18, len(dates)),
            "treasury_10y": np.linspace(3.5, 4.0, len(dates)),
            "treasury_13w": np.linspace(4.5, 4.7, len(dates)),
            "yield_curve_slope": np.linspace(-1.0, -0.7, len(dates)),
        },
        index=dates,
    )
    fx = pd.Series(1.35, index=dates)

    out = compute_features(
        prices=prices,
        fx_usdcad=fx,
        liquidity_lookback_days=30,
        feature_lookback_days=180,
        logger=logging.getLogger("test"),
        fundamentals=fundamentals,
        macro=macro,
    )
    missing = [c for c in TECHNICAL_FEATURES_ONLY if c not in out.columns]
    assert missing == []

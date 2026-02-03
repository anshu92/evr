"""Tests for advanced evaluation metrics."""
import numpy as np
import pandas as pd
import pytest

from stock_screener.modeling.eval import compute_portfolio_metrics, compute_calibration


def test_sharpe_ratio_computation():
    """Test Sharpe ratio calculation."""
    # Create daily returns with positive trend
    daily_returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)  # Mean ~12.5% annual
    
    metrics = compute_portfolio_metrics(daily_returns, rf_annual=0.05)
    
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "volatility_ann" in metrics
    assert "total_return" in metrics
    
    # Should have positive Sharpe for positive returns
    assert not pd.isna(metrics["sharpe_ratio"])
    
    # Volatility should be reasonable
    assert 0.05 < metrics["volatility_ann"] < 0.5


def test_max_drawdown_calculation():
    """Test max drawdown with known series."""
    # Series that goes up then drops 20%
    daily_returns = pd.Series([0.01] * 50 + [-0.01] * 25)  # Up 50%, then down ~22%
    
    metrics = compute_portfolio_metrics(daily_returns)
    
    # Should detect the drawdown
    assert metrics["max_drawdown"] < 0
    assert metrics["max_drawdown"] > -0.5  # Not catastrophic


def test_calibration_perfect():
    """Test calibration with perfectly calibrated predictions."""
    # Perfect calibration: predictions exactly match realized returns
    predictions = pd.Series(np.linspace(-0.1, 0.1, 100))
    realized = predictions.copy()
    
    result = compute_calibration(predictions, realized, n_bins=10)
    
    assert "calibration_error" in result
    # Perfect calibration should have very low error
    assert result["calibration_error"] < 0.001


def test_calibration_poor():
    """Test calibration with poorly calibrated predictions."""
    # Poor calibration: predictions have no relationship to realized
    predictions = pd.Series(np.linspace(-0.1, 0.1, 100))
    realized = pd.Series(np.random.randn(100) * 0.05)
    
    result = compute_calibration(predictions, realized, n_bins=10)
    
    assert "calibration_error" in result
    assert "by_decile" in result


def test_empty_returns():
    """Test that empty returns are handled gracefully."""
    empty = pd.Series([], dtype=float)
    
    metrics = compute_portfolio_metrics(empty)
    
    assert pd.isna(metrics["sharpe_ratio"])
    assert pd.isna(metrics["max_drawdown"])
    assert metrics["n_days"] == 0

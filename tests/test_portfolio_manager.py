import logging
from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

from stock_screener.portfolio.manager import PortfolioManager, TradeAction
from stock_screener.portfolio.state import PortfolioState, Position


def test_portfolio_manager_time_exit():
    """Test that positions exit at max_holding_days."""
    logger = logging.getLogger("test")
    
    manager = PortfolioManager(
        state_path="test_state.json",
        max_holding_days=5,
        max_holding_days_hard=10,
        extend_hold_min_pred_return=None,
        extend_hold_min_score=None,
        max_positions=5,
        stop_loss_pct=None,
        take_profit_pct=None,
        peak_detection_enabled=False,
        peak_sell_portion_pct=0.5,
        peak_min_gain_pct=None,
        peak_min_holding_days=2,
        peak_pred_return_threshold=None,
        peak_score_percentile_drop=None,
        peak_rsi_overbought=None,
        peak_above_ma_ratio=None,
        logger=logger,
    )
    
    # Create a position that's 6 days old
    now = datetime.now(tz=timezone.utc)
    entry_date = now - timedelta(days=6)
    
    state = PortfolioState(cash_cad=1000.0, positions=[])
    state.positions.append(
        Position(
            ticker="AAPL",
            entry_price=100.0,
            entry_date=entry_date,
            shares=10,
        )
    )
    
    prices = pd.Series({"AAPL": 105.0})
    
    actions = manager.apply_exits(state, prices)
    
    # Should have one TIME_EXIT action
    assert len(actions) == 1
    assert actions[0].action == "SELL"
    assert actions[0].reason == "TIME_EXIT"
    assert actions[0].ticker == "AAPL"


def test_portfolio_manager_stop_loss():
    """Test that stop loss triggers correctly."""
    logger = logging.getLogger("test")
    
    manager = PortfolioManager(
        state_path="test_state.json",
        max_holding_days=10,
        max_holding_days_hard=20,
        extend_hold_min_pred_return=None,
        extend_hold_min_score=None,
        max_positions=5,
        stop_loss_pct=0.10,  # 10% stop loss
        take_profit_pct=None,
        peak_detection_enabled=False,
        peak_sell_portion_pct=0.5,
        peak_min_gain_pct=None,
        peak_min_holding_days=2,
        peak_pred_return_threshold=None,
        peak_score_percentile_drop=None,
        peak_rsi_overbought=None,
        peak_above_ma_ratio=None,
        logger=logger,
    )
    
    # Create a position that's down 11%
    now = datetime.now(tz=timezone.utc)
    entry_date = now - timedelta(days=2)
    
    state = PortfolioState(cash_cad=1000.0, positions=[])
    state.positions.append(
        Position(
            ticker="AAPL",
            entry_price=100.0,
            entry_date=entry_date,
            shares=10,
        )
    )
    
    prices = pd.Series({"AAPL": 89.0})  # Down 11%
    
    actions = manager.apply_exits(state, prices)
    
    # Should have one STOP_LOSS action
    assert len(actions) == 1
    assert actions[0].action == "SELL"
    assert actions[0].reason == "STOP_LOSS"


def test_portfolio_manager_take_profit():
    """Test that take profit triggers correctly."""
    logger = logging.getLogger("test")
    
    manager = PortfolioManager(
        state_path="test_state.json",
        max_holding_days=10,
        max_holding_days_hard=20,
        extend_hold_min_pred_return=None,
        extend_hold_min_score=None,
        max_positions=5,
        stop_loss_pct=None,
        take_profit_pct=0.15,  # 15% take profit
        peak_detection_enabled=False,
        peak_sell_portion_pct=0.5,
        peak_min_gain_pct=None,
        peak_min_holding_days=2,
        peak_pred_return_threshold=None,
        peak_score_percentile_drop=None,
        peak_rsi_overbought=None,
        peak_above_ma_ratio=None,
        logger=logger,
    )
    
    # Create a position that's up 16%
    now = datetime.now(tz=timezone.utc)
    entry_date = now - timedelta(days=2)
    
    state = PortfolioState(cash_cad=1000.0, positions=[])
    state.positions.append(
        Position(
            ticker="AAPL",
            entry_price=100.0,
            entry_date=entry_date,
            shares=10,
        )
    )
    
    prices = pd.Series({"AAPL": 116.0})  # Up 16%
    
    actions = manager.apply_exits(state, prices)
    
    # Should have one TAKE_PROFIT action
    assert len(actions) == 1
    assert actions[0].action == "SELL"
    assert actions[0].reason == "TAKE_PROFIT"


def test_portfolio_manager_peak_detection():
    """Test that peak detection triggers partial exit."""
    logger = logging.getLogger("test")
    
    manager = PortfolioManager(
        state_path="test_state.json",
        max_holding_days=10,
        max_holding_days_hard=20,
        extend_hold_min_pred_return=None,
        extend_hold_min_score=None,
        max_positions=5,
        stop_loss_pct=None,
        take_profit_pct=None,
        peak_detection_enabled=True,
        peak_sell_portion_pct=0.5,  # Sell 50%
        peak_min_gain_pct=0.05,  # Need at least 5% gain
        peak_min_holding_days=2,
        peak_pred_return_threshold=-0.02,  # Negative prediction
        peak_score_percentile_drop=None,
        peak_rsi_overbought=70.0,
        peak_above_ma_ratio=None,
        logger=logger,
    )
    
    # Create a position that's up 10% and held for 3 days
    now = datetime.now(tz=timezone.utc)
    entry_date = now - timedelta(days=3)
    
    state = PortfolioState(cash_cad=1000.0, positions=[])
    state.positions.append(
        Position(
            ticker="AAPL",
            entry_price=100.0,
            entry_date=entry_date,
            shares=10,
        )
    )
    
    prices = pd.Series({"AAPL": 110.0})  # Up 10%
    
    # Features with peak signals
    features = pd.DataFrame({
        "rsi_14": [75.0],  # Overbought
    }, index=["AAPL"])
    
    pred_return = pd.Series({"AAPL": -0.03})  # Negative prediction
    
    actions = manager.apply_exits(state, prices, pred_return=pred_return, features=features)
    
    # Should have one SELL_PARTIAL action (2 peak signals: RSI + neg prediction)
    assert len(actions) == 1
    assert actions[0].action == "SELL_PARTIAL"
    assert "PEAK" in actions[0].reason
    assert actions[0].shares == 5  # 50% of 10

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
        peak_based_exit=False,
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
    
    # Create a position old enough to exceed 5 trading days.
    now = datetime.now(tz=timezone.utc)
    entry_date = now - timedelta(days=8)
    
    state = PortfolioState(cash_cad=1000.0, positions=[], last_updated=now)
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
        peak_based_exit=False,
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
    
    state = PortfolioState(cash_cad=1000.0, positions=[], last_updated=now)
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
        peak_based_exit=False,
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
    
    state = PortfolioState(cash_cad=1000.0, positions=[], last_updated=now)
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
        peak_based_exit=False,
        twr_optimization=False,
        signal_decay_exit_enabled=False,
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
    
    # Create a position that's up 10% and held long enough in trading days.
    now = datetime.now(tz=timezone.utc)
    entry_date = now - timedelta(days=7)
    
    state = PortfolioState(cash_cad=1000.0, positions=[], last_updated=now)
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


def test_fractional_buy_when_cash_below_share_price(tmp_path):
    """Allow fractional entry even when cash is below one full share price."""
    logger = logging.getLogger("test")
    now = datetime.now(tz=timezone.utc)

    manager = PortfolioManager(
        state_path=str(tmp_path / "state.json"),
        max_holding_days=5,
        max_holding_days_hard=10,
        extend_hold_min_pred_return=None,
        extend_hold_min_score=None,
        max_positions=1,
        stop_loss_pct=None,
        take_profit_pct=None,
        peak_based_exit=False,
        peak_detection_enabled=False,
        peak_sell_portion_pct=0.5,
        peak_min_gain_pct=None,
        peak_min_holding_days=2,
        peak_pred_return_threshold=None,
        peak_score_percentile_drop=None,
        peak_rsi_overbought=None,
        peak_above_ma_ratio=None,
        min_trade_notional_cad=10.0,
        logger=logger,
    )

    state = PortfolioState(cash_cad=100.0, positions=[], last_updated=now)
    screened = pd.DataFrame(index=["AAPL"])
    weights = pd.DataFrame({"weight": [1.0]}, index=["AAPL"])
    prices = pd.Series({"AAPL": 300.0})

    plan = manager.build_trade_plan(
        state=state,
        screened=screened,
        weights=weights,
        prices_cad=prices,
    )
    buys = [a for a in plan.actions if a.action == "BUY" and a.ticker == "AAPL"]
    assert len(buys) == 1
    assert buys[0].shares > 0
    assert buys[0].shares < 1.0


def test_rotation_max_days_uses_trading_days(monkeypatch, tmp_path):
    """Do not rotate purely on weekend calendar drift; use trading days."""
    logger = logging.getLogger("test")
    now = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)  # Monday
    entry_date = datetime(2025, 1, 3, 15, 0, tzinfo=timezone.utc)  # Friday

    monkeypatch.setattr("stock_screener.portfolio.manager._utcnow", lambda: now)

    manager = PortfolioManager(
        state_path=str(tmp_path / "state.json"),
        max_holding_days=1,
        max_holding_days_hard=2,
        extend_hold_min_pred_return=None,
        extend_hold_min_score=None,
        max_positions=5,
        stop_loss_pct=None,
        take_profit_pct=None,
        peak_based_exit=False,
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

    state = PortfolioState(
        cash_cad=1000.0,
        positions=[Position(ticker="AAPL", entry_price=100.0, entry_date=entry_date, shares=1.0)],
        last_updated=now,
    )
    screened = pd.DataFrame({"pred_return": [0.01]}, index=["AAPL"])
    weights = pd.DataFrame({"weight": []})
    prices = pd.Series({"AAPL": 102.0})

    plan = manager.build_trade_plan(
        state=state,
        screened=screened,
        weights=weights,
        prices_cad=prices,
    )

    sells = [a for a in plan.actions if a.action == "SELL" and a.ticker == "AAPL"]
    assert len(sells) == 0


def test_holdings_days_held_reports_trading_days(monkeypatch, tmp_path):
    """Holdings summary should show trading days, not calendar days."""
    logger = logging.getLogger("test")
    now = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)  # Monday
    entry_date = datetime(2025, 1, 3, 15, 0, tzinfo=timezone.utc)  # Friday

    monkeypatch.setattr("stock_screener.portfolio.manager._utcnow", lambda: now)

    manager = PortfolioManager(
        state_path=str(tmp_path / "state.json"),
        max_holding_days=5,
        max_holding_days_hard=10,
        extend_hold_min_pred_return=None,
        extend_hold_min_score=None,
        max_positions=5,
        stop_loss_pct=None,
        take_profit_pct=None,
        peak_based_exit=False,
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

    state = PortfolioState(
        cash_cad=1000.0,
        positions=[Position(ticker="AAPL", entry_price=100.0, entry_date=entry_date, shares=1.0)],
        last_updated=now,
    )
    screened = pd.DataFrame({"pred_return": [0.01]}, index=["AAPL"])
    weights = pd.DataFrame({"weight": [1.0]}, index=["AAPL"])
    prices = pd.Series({"AAPL": 102.0})

    plan = manager.build_trade_plan(
        state=state,
        screened=screened,
        weights=weights,
        prices_cad=prices,
    )

    assert int(plan.holdings.loc["AAPL", "days_held"]) == 1


def test_cold_start_seed_buy_allowed_below_min_trade_notional(tmp_path):
    logger = logging.getLogger("test")
    now = datetime.now(tz=timezone.utc)

    manager = PortfolioManager(
        state_path=str(tmp_path / "state.json"),
        max_holding_days=5,
        max_holding_days_hard=10,
        extend_hold_min_pred_return=None,
        extend_hold_min_score=None,
        max_positions=1,
        stop_loss_pct=None,
        take_profit_pct=None,
        peak_based_exit=False,
        peak_detection_enabled=False,
        peak_sell_portion_pct=0.5,
        peak_min_gain_pct=None,
        peak_min_holding_days=2,
        peak_pred_return_threshold=None,
        peak_score_percentile_drop=None,
        peak_rsi_overbought=None,
        peak_above_ma_ratio=None,
        min_trade_notional_cad=15.0,
        min_rebalance_weight_delta=0.015,
        logger=logger,
    )

    state = PortfolioState(cash_cad=427.49, positions=[], last_updated=now)
    screened = pd.DataFrame(index=["AAA"])
    weights = pd.DataFrame({"weight": [0.01975]}, index=["AAA"])
    prices = pd.Series({"AAA": 100.0})

    plan = manager.build_trade_plan(
        state=state,
        screened=screened,
        weights=weights,
        prices_cad=prices,
    )
    buys = [a for a in plan.actions if a.action == "BUY" and a.ticker == "AAA"]
    assert len(buys) == 1
    assert buys[0].reason == "TOP_RANKED:SEED_NOTIONAL"
    assert 1.0 <= float(buys[0].shares * buys[0].price_cad) < 15.0

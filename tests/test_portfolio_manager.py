import logging
from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

from stock_screener.portfolio.manager import (
    PortfolioManager,
    TradeAction,
    _trading_days_between,
)
from stock_screener.portfolio.state import PortfolioState, Position


def test_portfolio_manager_no_time_exit():
    """Positions should not be sold purely for exceeding holding days."""
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
    
    # No max-days sell logic: this should remain open.
    assert len(actions) == 0
    assert len([p for p in state.positions if p.status == "OPEN" and p.ticker == "AAPL"]) == 1


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
    assert actions[0].entry_price == pytest.approx(100.0)
    assert actions[0].realized_gain_pct == pytest.approx(0.10)


def test_peak_full_liquidation_emits_sell(tmp_path):
    """A 100% peak sell should be treated as a full SELL, not SELL_PARTIAL."""
    logger = logging.getLogger("test")
    now = datetime.now(tz=timezone.utc)
    entry_date = now - timedelta(days=7)

    manager = PortfolioManager(
        state_path=str(tmp_path / "state.json"),
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
        peak_sell_portion_pct=1.0,  # full liquidation
        peak_min_gain_pct=0.05,
        peak_min_holding_days=2,
        peak_pred_return_threshold=-0.02,
        peak_score_percentile_drop=None,
        peak_rsi_overbought=70.0,
        peak_above_ma_ratio=None,
        logger=logger,
    )

    state = PortfolioState(
        cash_cad=0.0,
        positions=[Position(ticker="AAPL", entry_price=100.0, entry_date=entry_date, shares=2.0)],
        last_updated=now,
    )
    prices = pd.Series({"AAPL": 110.0})
    features = pd.DataFrame({"rsi_14": [75.0]}, index=["AAPL"])
    pred_return = pd.Series({"AAPL": -0.03})

    actions = manager.apply_exits(state, prices, pred_return=pred_return, features=features)

    assert len(actions) == 1
    assert actions[0].action == "SELL"
    assert actions[0].reason.startswith("PEAK_")
    assert len([p for p in state.positions if p.status == "OPEN"]) == 0


def test_peak_detection_allows_fractional_partial_sell(tmp_path):
    """Peak partial exits should work for fractional positions below 2 shares."""
    logger = logging.getLogger("test")
    now = datetime.now(tz=timezone.utc)
    entry_date = now - timedelta(days=7)

    manager = PortfolioManager(
        state_path=str(tmp_path / "state.json"),
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
        peak_sell_portion_pct=0.5,
        peak_min_gain_pct=0.05,
        peak_min_holding_days=2,
        peak_pred_return_threshold=-0.02,
        peak_score_percentile_drop=None,
        peak_rsi_overbought=70.0,
        peak_above_ma_ratio=None,
        logger=logger,
    )

    state = PortfolioState(
        cash_cad=0.0,
        positions=[Position(ticker="AAPL", entry_price=100.0, entry_date=entry_date, shares=1.5)],
        last_updated=now,
    )
    prices = pd.Series({"AAPL": 110.0})
    features = pd.DataFrame({"rsi_14": [75.0]}, index=["AAPL"])
    pred_return = pd.Series({"AAPL": -0.03})

    actions = manager.apply_exits(state, prices, pred_return=pred_return, features=features)
    assert len(actions) == 1
    assert actions[0].action == "SELL_PARTIAL"
    assert actions[0].shares == pytest.approx(0.75)
    open_positions = [p for p in state.positions if p.status == "OPEN" and p.ticker == "AAPL"]
    assert len(open_positions) == 1
    assert open_positions[0].shares == pytest.approx(0.75)


def test_peak_partial_sell_respects_same_day_cooldown(monkeypatch, tmp_path):
    """Do not repeatedly partial-sell the same position in same trading day."""
    logger = logging.getLogger("test")
    now = datetime(2025, 1, 10, 15, 0, tzinfo=timezone.utc)
    entry_date = datetime(2025, 1, 3, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("stock_screener.portfolio.manager._utcnow", lambda: now)

    manager = PortfolioManager(
        state_path=str(tmp_path / "state.json"),
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
        peak_sell_portion_pct=0.5,
        peak_min_gain_pct=0.05,
        peak_min_holding_days=2,
        peak_pred_return_threshold=-0.02,
        peak_score_percentile_drop=None,
        peak_rsi_overbought=70.0,
        peak_above_ma_ratio=None,
        logger=logger,
    )

    state = PortfolioState(
        cash_cad=0.0,
        positions=[
            Position(
                ticker="AAPL",
                entry_price=100.0,
                entry_date=entry_date,
                shares=2.0,
                last_partial_sell_at=now,  # same-day prior partial sell
            )
        ],
        last_updated=now,
    )
    prices = pd.Series({"AAPL": 110.0})
    features = pd.DataFrame({"rsi_14": [75.0]}, index=["AAPL"])
    pred_return = pd.Series({"AAPL": -0.03})

    actions = manager.apply_exits(state, prices, pred_return=pred_return, features=features)
    assert actions == []
    assert state.positions[0].shares == pytest.approx(2.0)


def test_build_trade_plan_consolidates_duplicate_open_lots(tmp_path):
    """Duplicate open lots for the same ticker should be consolidated deterministically."""
    logger = logging.getLogger("test")
    now = datetime.now(tz=timezone.utc)
    state = PortfolioState(
        cash_cad=0.0,
        positions=[
            Position(ticker="AAPL", entry_price=100.0, entry_date=now - timedelta(days=5), shares=1.0),
            Position(ticker="AAPL", entry_price=110.0, entry_date=now - timedelta(days=3), shares=2.0),
        ],
        last_updated=now,
    )
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

    screened = pd.DataFrame({"pred_return": [0.02], "score": [1.0]}, index=["AAPL"])
    weights = pd.DataFrame({"weight": [1.0], "pred_return": [0.02]}, index=["AAPL"])
    prices = pd.Series({"AAPL": 120.0})
    plan = manager.build_trade_plan(
        state=state,
        screened=screened,
        weights=weights,
        prices_cad=prices,
    )

    holds = [a for a in plan.actions if a.action == "HOLD" and a.ticker == "AAPL"]
    assert len(holds) == 1
    assert holds[0].shares == pytest.approx(3.0)
    assert len([p for p in state.positions if p.status == "OPEN" and p.ticker == "AAPL"]) == 1


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


def test_rotation_does_not_force_sell_on_holding_days(monkeypatch, tmp_path):
    """Rotation should not force sell solely because a position is old."""
    logger = logging.getLogger("test")
    now = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)  # Monday
    entry_date = datetime(2024, 12, 16, 15, 0, tzinfo=timezone.utc)  # Older holding

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
    screened = pd.DataFrame(
        {"pred_return": [0.02, 0.01, 0.00, -0.01], "score": [0.95, 0.85, 0.75, 0.65]},
        index=["AAPL", "BBB", "CCC", "DDD"],
    )
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


def test_blocked_buys_prevents_same_run_reentry(tmp_path):
    """Tickers exited earlier in the run must not be re-bought in the same plan."""
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
        logger=logger,
    )

    state = PortfolioState(cash_cad=1000.0, positions=[], last_updated=now)
    screened = pd.DataFrame({"pred_return": [0.05]}, index=["AAPL"])
    weights = pd.DataFrame({"weight": [1.0], "pred_return": [0.05]}, index=["AAPL"])
    prices = pd.Series({"AAPL": 200.0})

    plan = manager.build_trade_plan(
        state=state,
        screened=screened,
        weights=weights,
        prices_cad=prices,
        blocked_buys={"aapl"},
    )

    assert len([a for a in plan.actions if a.action == "BUY" and a.ticker == "AAPL"]) == 0
    assert len([p for p in state.positions if p.status == "OPEN" and p.ticker == "AAPL"]) == 0
    assert state.cash_cad == pytest.approx(1000.0)


def test_rotation_buy_records_replaced_ticker(monkeypatch, tmp_path):
    """BUY actions should carry explicit replaced ticker for reward attribution."""
    logger = logging.getLogger("test")
    now = datetime(2025, 1, 10, 15, 0, tzinfo=timezone.utc)
    entry_date = datetime(2025, 1, 3, 15, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("stock_screener.portfolio.manager._utcnow", lambda: now)

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
        rotate_on_missing_data=True,
        logger=logger,
    )

    state = PortfolioState(
        cash_cad=0.0,
        positions=[Position(ticker="OLD", entry_price=100.0, entry_date=entry_date, shares=1.0)],
        last_updated=now,
    )
    screened = pd.DataFrame({"pred_return": [0.02]}, index=["NEW"])
    weights = pd.DataFrame({"weight": [1.0], "pred_return": [0.02]}, index=["NEW"])
    prices = pd.Series({"OLD": 100.0, "NEW": 100.0})

    plan = manager.build_trade_plan(
        state=state,
        screened=screened,
        weights=weights,
        prices_cad=prices,
    )

    buys = [a for a in plan.actions if a.action == "BUY" and a.ticker == "NEW"]
    sells = [a for a in plan.actions if a.action == "SELL" and a.ticker == "OLD"]
    assert len(buys) == 1
    assert len(sells) == 1
    assert buys[0].replaces_ticker == "OLD"


def test_rotation_does_not_sell_missing_data_by_default(monkeypatch, tmp_path):
    """Holdings absent from screened should not be auto-rotated by missing data alone."""
    logger = logging.getLogger("test")
    now = datetime(2025, 1, 10, 15, 0, tzinfo=timezone.utc)
    entry_date = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("stock_screener.portfolio.manager._utcnow", lambda: now)

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
        logger=logger,
    )

    state = PortfolioState(
        cash_cad=0.0,
        positions=[Position(ticker="OLD", entry_price=100.0, entry_date=entry_date, shares=1.0)],
        last_updated=now,
    )
    screened = pd.DataFrame({"pred_return": [0.02], "score": [0.9]}, index=["NEW"])
    weights = pd.DataFrame({"weight": [1.0], "pred_return": [0.02]}, index=["NEW"])
    prices = pd.Series({"OLD": 101.0, "NEW": 100.0})

    plan = manager.build_trade_plan(
        state=state,
        screened=screened,
        weights=weights,
        prices_cad=prices,
    )

    assert len([a for a in plan.actions if a.action == "SELL" and a.ticker == "OLD"]) == 0
    assert len([a for a in plan.actions if a.action == "HOLD" and a.ticker == "OLD"]) == 1


def test_rotation_uses_feature_prediction_for_out_of_screen_holdings(monkeypatch, tmp_path):
    """Out-of-screen holdings should rotate on negative live prediction from features."""
    logger = logging.getLogger("test")
    now = datetime(2025, 1, 10, 15, 0, tzinfo=timezone.utc)
    entry_date = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("stock_screener.portfolio.manager._utcnow", lambda: now)

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
        logger=logger,
    )

    state = PortfolioState(
        cash_cad=0.0,
        positions=[Position(ticker="OLD", entry_price=100.0, entry_date=entry_date, shares=1.0)],
        last_updated=now,
    )
    # OLD is not in screened/top-N set.
    screened = pd.DataFrame({"pred_return": [0.02], "score": [0.9]}, index=["NEW"])
    weights = pd.DataFrame({"weight": [1.0], "pred_return": [0.02]}, index=["NEW"])
    prices = pd.Series({"OLD": 101.0, "NEW": 100.0})
    features = pd.DataFrame({"pred_return": [-0.05]}, index=["OLD"])

    plan = manager.build_trade_plan(
        state=state,
        screened=screened,
        weights=weights,
        prices_cad=prices,
        features=features,
    )

    sells = [a for a in plan.actions if a.action == "SELL" and a.ticker == "OLD"]
    assert len(sells) == 1
    assert sells[0].reason == "ROTATION:NEG_PRED"


def test_trading_days_between_uses_market_calendar():
    """US and CA calendars should differ on market-specific holidays."""
    start = datetime(2025, 8, 1, 15, 0, tzinfo=timezone.utc)  # Friday
    end = datetime(2025, 8, 4, 15, 0, tzinfo=timezone.utc)  # Monday (Civic Holiday in CA)

    assert _trading_days_between(start, end, market="US") == 1
    assert _trading_days_between(start, end, market="CA") == 0

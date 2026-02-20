import logging
from datetime import datetime, timezone

import pandas as pd
import pytest

from stock_screener.pipeline.daily import _apply_rebalance_controls
from stock_screener.portfolio.state import PortfolioState, Position


def _build_state() -> PortfolioState:
    now = datetime.now(tz=timezone.utc)
    return PortfolioState(
        cash_cad=0.0,
        positions=[
            Position(
                ticker="AAA",
                entry_price=100.0,
                entry_date=now,
                shares=10.0,  # 1000 CAD market value
            )
        ],
        last_updated=now,
    )


def test_dynamic_no_trade_band_blocks_small_high_friction_changes():
    logger = logging.getLogger("test")
    state = _build_state()
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0})
    target = pd.DataFrame({"weight": [0.98, 0.02]}, index=["AAA", "BBB"])
    screened = pd.DataFrame(
        {
            "pred_uncertainty": [0.05, 0.90],
            "avg_dollar_volume_cad": [20_000_000.0, 50_000.0],
        },
        index=["AAA", "BBB"],
    )

    static_out = _apply_rebalance_controls(
        target,
        state=state,
        screened=screened,
        prices_cad=prices,
        market_vol_regime=1.8,
        min_rebalance_weight_delta=0.01,
        min_trade_notional_cad=10.0,
        turnover_penalty_bps=10.0,
        dynamic_band_enabled=False,
        uncertainty_weight=0.8,
        liquidity_weight=0.6,
        vol_regime_weight=0.5,
        band_mult_min=0.75,
        band_mult_max=2.5,
        logger=logger,
    )
    assert "BBB" in static_out.index
    assert float(static_out.loc["BBB", "weight"]) > 0.0

    dynamic_out = _apply_rebalance_controls(
        target,
        state=state,
        screened=screened,
        prices_cad=prices,
        market_vol_regime=1.8,
        min_rebalance_weight_delta=0.01,
        min_trade_notional_cad=10.0,
        turnover_penalty_bps=10.0,
        dynamic_band_enabled=True,
        uncertainty_weight=0.8,
        liquidity_weight=0.6,
        vol_regime_weight=0.5,
        band_mult_min=0.75,
        band_mult_max=2.5,
        logger=logger,
    )
    assert "BBB" not in dynamic_out.index or float(dynamic_out.loc["BBB", "weight"]) == 0.0


def test_rebalance_controls_allow_cold_start_entries():
    logger = logging.getLogger("test")
    state = PortfolioState(
        cash_cad=450.0,
        positions=[],
        last_updated=datetime.now(tz=timezone.utc),
    )
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 100.0})
    target = pd.DataFrame({"weight": [0.40, 0.35, 0.25]}, index=["AAA", "BBB", "CCC"])
    screened = pd.DataFrame(
        {
            "pred_uncertainty": [0.95, 0.90, 0.85],
            "avg_dollar_volume_cad": [50_000.0, 80_000.0, 120_000.0],
        },
        index=["AAA", "BBB", "CCC"],
    )

    out = _apply_rebalance_controls(
        target,
        state=state,
        screened=screened,
        prices_cad=prices,
        market_vol_regime=1.9,
        min_rebalance_weight_delta=0.015,
        min_trade_notional_cad=15.0,
        turnover_penalty_bps=15.0,
        dynamic_band_enabled=True,
        uncertainty_weight=1.2,
        liquidity_weight=0.8,
        vol_regime_weight=0.8,
        band_mult_min=1.0,
        band_mult_max=3.0,
        logger=logger,
    )
    assert not out.empty
    assert float(out["weight"].sum()) > 0.0


def test_rebalance_controls_keep_seed_target_below_min_notional():
    logger = logging.getLogger("test")
    state = PortfolioState(
        cash_cad=427.49,
        positions=[],
        last_updated=datetime.now(tz=timezone.utc),
    )
    prices = pd.Series({"AAA": 100.0})
    target = pd.DataFrame({"weight": [0.01975]}, index=["AAA"])
    screened = pd.DataFrame(
        {
            "pred_uncertainty": [0.9],
            "avg_dollar_volume_cad": [100_000.0],
        },
        index=["AAA"],
    )

    out = _apply_rebalance_controls(
        target,
        state=state,
        screened=screened,
        prices_cad=prices,
        market_vol_regime=2.0,
        min_rebalance_weight_delta=0.015,
        min_trade_notional_cad=15.0,
        turnover_penalty_bps=15.0,
        dynamic_band_enabled=True,
        uncertainty_weight=1.2,
        liquidity_weight=0.8,
        vol_regime_weight=0.8,
        band_mult_min=1.0,
        band_mult_max=3.0,
        logger=logger,
    )
    assert not out.empty
    assert "AAA" in out.index
    assert float(out.loc["AAA", "weight"]) > 0.0


def test_rebalance_controls_can_disable_turnover_shrinkage():
    logger = logging.getLogger("test")
    state = _build_state()
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0})
    target = pd.DataFrame({"weight": [0.50, 0.50]}, index=["AAA", "BBB"])
    screened = pd.DataFrame(index=["AAA", "BBB"])

    with_shrink = _apply_rebalance_controls(
        target,
        state=state,
        screened=screened,
        prices_cad=prices,
        market_vol_regime=1.0,
        min_rebalance_weight_delta=0.0,
        min_trade_notional_cad=1.0,
        turnover_penalty_bps=100.0,
        dynamic_band_enabled=False,
        uncertainty_weight=0.0,
        liquidity_weight=0.0,
        vol_regime_weight=0.0,
        band_mult_min=1.0,
        band_mult_max=1.0,
        logger=logger,
        apply_turnover_shrinkage=True,
    )
    without_shrink = _apply_rebalance_controls(
        target,
        state=state,
        screened=screened,
        prices_cad=prices,
        market_vol_regime=1.0,
        min_rebalance_weight_delta=0.0,
        min_trade_notional_cad=1.0,
        turnover_penalty_bps=100.0,
        dynamic_band_enabled=False,
        uncertainty_weight=0.0,
        liquidity_weight=0.0,
        vol_regime_weight=0.0,
        band_mult_min=1.0,
        band_mult_max=1.0,
        logger=logger,
        apply_turnover_shrinkage=False,
    )

    assert float(with_shrink.loc["BBB", "weight"]) < float(without_shrink.loc["BBB", "weight"])
    assert float(without_shrink.loc["BBB", "weight"]) == pytest.approx(0.50, abs=1e-9)


def test_rebalance_controls_allow_cash_policy_does_not_upscale_underinvested_weights():
    logger = logging.getLogger("test")
    state = PortfolioState(
        cash_cad=450.0,
        positions=[],
        last_updated=datetime.now(tz=timezone.utc),
    )
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0})
    target = pd.DataFrame({"weight": [0.30, 0.20]}, index=["AAA", "BBB"])
    screened = pd.DataFrame(index=["AAA", "BBB"])

    out = _apply_rebalance_controls(
        target,
        state=state,
        screened=screened,
        prices_cad=prices,
        market_vol_regime=1.0,
        min_rebalance_weight_delta=0.0,
        min_trade_notional_cad=1.0,
        turnover_penalty_bps=0.0,
        dynamic_band_enabled=False,
        uncertainty_weight=0.0,
        liquidity_weight=0.0,
        vol_regime_weight=0.0,
        band_mult_min=1.0,
        band_mult_max=1.0,
        logger=logger,
        exposure_policy="allow_cash_no_upscale",
        target_gross_exposure=1.0,
        allow_leverage=False,
    )
    assert float(out["weight"].sum()) == pytest.approx(0.50, abs=1e-9)


def test_rebalance_controls_can_normalize_to_target_gross_when_configured():
    logger = logging.getLogger("test")
    state = PortfolioState(
        cash_cad=450.0,
        positions=[],
        last_updated=datetime.now(tz=timezone.utc),
    )
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0})
    target = pd.DataFrame({"weight": [0.30, 0.20]}, index=["AAA", "BBB"])
    screened = pd.DataFrame(index=["AAA", "BBB"])

    out = _apply_rebalance_controls(
        target,
        state=state,
        screened=screened,
        prices_cad=prices,
        market_vol_regime=1.0,
        min_rebalance_weight_delta=0.0,
        min_trade_notional_cad=1.0,
        turnover_penalty_bps=0.0,
        dynamic_band_enabled=False,
        uncertainty_weight=0.0,
        liquidity_weight=0.0,
        vol_regime_weight=0.0,
        band_mult_min=1.0,
        band_mult_max=1.0,
        logger=logger,
        exposure_policy="normalize_to_target_gross",
        target_gross_exposure=1.0,
        allow_leverage=False,
    )
    assert float(out["weight"].sum()) == pytest.approx(1.0, abs=1e-9)


def test_rebalance_controls_normalize_policy_respects_no_leverage_cap():
    logger = logging.getLogger("test")
    state = PortfolioState(
        cash_cad=450.0,
        positions=[],
        last_updated=datetime.now(tz=timezone.utc),
    )
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0})
    target = pd.DataFrame({"weight": [0.30, 0.20]}, index=["AAA", "BBB"])
    screened = pd.DataFrame(index=["AAA", "BBB"])

    out = _apply_rebalance_controls(
        target,
        state=state,
        screened=screened,
        prices_cad=prices,
        market_vol_regime=1.0,
        min_rebalance_weight_delta=0.0,
        min_trade_notional_cad=1.0,
        turnover_penalty_bps=0.0,
        dynamic_band_enabled=False,
        uncertainty_weight=0.0,
        liquidity_weight=0.0,
        vol_regime_weight=0.0,
        band_mult_min=1.0,
        band_mult_max=1.0,
        logger=logger,
        exposure_policy="normalize_to_target_gross",
        target_gross_exposure=1.2,
        allow_leverage=False,
    )
    assert float(out["weight"].sum()) == pytest.approx(1.0, abs=1e-9)


def test_rebalance_controls_emits_notional_drop_reason_codes_in_diagnostics():
    logger = logging.getLogger("test")
    state = PortfolioState(
        cash_cad=100.0,
        positions=[],
        last_updated=datetime.now(tz=timezone.utc),
    )
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0})
    target = pd.DataFrame({"weight": [0.05, 0.95]}, index=["AAA", "BBB"])
    screened = pd.DataFrame(index=["AAA", "BBB"])

    out, diag = _apply_rebalance_controls(
        target,
        state=state,
        screened=screened,
        prices_cad=prices,
        market_vol_regime=1.0,
        min_rebalance_weight_delta=0.0,
        min_trade_notional_cad=20.0,
        turnover_penalty_bps=0.0,
        dynamic_band_enabled=False,
        uncertainty_weight=0.0,
        liquidity_weight=0.0,
        vol_regime_weight=0.0,
        band_mult_min=1.0,
        band_mult_max=1.0,
        logger=logger,
        return_diagnostics=True,
    )
    assert isinstance(out, pd.DataFrame)
    assert isinstance(diag, dict)
    assert diag.get("path") in {"cold_start", "cold_start_seed"}
    drops = [d for d in diag.get("dropped_by_notional_gate", []) if isinstance(d, dict)]
    assert any(str(d.get("ticker")) == "AAA" and str(d.get("reason_code")) == "cold_start_min_notional" for d in drops)


def test_rebalance_controls_reports_hysteresis_notional_drop_in_diagnostics():
    logger = logging.getLogger("test")
    state = _build_state()  # ~1000 CAD in AAA, no cash
    prices = pd.Series({"AAA": 100.0, "BBB": 100.0})
    target = pd.DataFrame({"weight": [0.99, 0.01]}, index=["AAA", "BBB"])
    screened = pd.DataFrame(index=["AAA", "BBB"])

    out, diag = _apply_rebalance_controls(
        target,
        state=state,
        screened=screened,
        prices_cad=prices,
        market_vol_regime=1.0,
        min_rebalance_weight_delta=0.0,
        min_trade_notional_cad=20.0,
        turnover_penalty_bps=0.0,
        dynamic_band_enabled=False,
        uncertainty_weight=0.0,
        liquidity_weight=0.0,
        vol_regime_weight=0.0,
        band_mult_min=1.0,
        band_mult_max=1.0,
        logger=logger,
        return_diagnostics=True,
    )
    assert isinstance(out, pd.DataFrame)
    assert isinstance(diag, dict)
    drops = [d for d in diag.get("dropped_by_notional_gate", []) if isinstance(d, dict)]
    assert any(str(d.get("ticker")) == "BBB" and str(d.get("reason_code")) == "hysteresis_notional" for d in drops)

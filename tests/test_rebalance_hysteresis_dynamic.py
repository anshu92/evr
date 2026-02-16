import logging
from datetime import datetime, timezone

import pandas as pd

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

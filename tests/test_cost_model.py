from __future__ import annotations

import pandas as pd
import pytest

from stock_screener.config import Config
from stock_screener.modeling.costs import apply_cost_to_label, estimate_trade_cost_bps


def test_apply_cost_to_label_converts_gross_to_net():
    gross = pd.Series([0.05, 0.02, -0.01])
    cost_bps = pd.Series([10.0, 25.0, 5.0])
    net = apply_cost_to_label(gross, cost_bps)
    assert net.tolist() == pytest.approx([0.049, 0.0175, -0.0105])


def test_estimate_trade_cost_bps_penalizes_illiquidity_and_volatility():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"] * 4),
            "avg_dollar_volume_cad": [20_000_000, 2_000_000, 20_000_000, 2_000_000],
            "vol_20d_ann": [0.10, 0.10, 0.60, 0.60],
        },
        index=["LIQ_LOWVOL", "ILLIQ_LOWVOL", "LIQ_HIGHVOL", "ILLIQ_HIGHVOL"],
    )
    costs = estimate_trade_cost_bps(
        df,
        base_bps=10.0,
        spread_coef=1.0,
        vol_coef=1.0,
        min_bps=0.0,
        max_bps=200.0,
    )
    assert costs["ILLIQ_HIGHVOL"] > costs["LIQ_HIGHVOL"] > costs["LIQ_LOWVOL"]
    assert costs["ILLIQ_HIGHVOL"] > costs["ILLIQ_LOWVOL"] > costs["LIQ_LOWVOL"]


def test_config_parses_cost_model_env(monkeypatch):
    monkeypatch.setenv("COST_MODEL_BASE_BPS", "4.5")
    monkeypatch.setenv("COST_MODEL_SPREAD_COEF", "0.7")
    monkeypatch.setenv("COST_MODEL_VOL_COEF", "0.9")
    cfg = Config.from_env()
    assert cfg.cost_model_base_bps == 4.5
    assert cfg.cost_model_spread_coef == 0.7
    assert cfg.cost_model_vol_coef == 0.9

import logging
import numpy as np

import pandas as pd

from stock_screener.optimization.risk_parity import (
    apply_correlation_limits,
    apply_max_position_cap,
    compute_inverse_vol_weights,
)


def test_weight_cap_allows_cash_when_infeasible():
    logger = logging.getLogger("test")
    df = pd.DataFrame(
        {
            "vol_60d_ann": [0.2, 0.2, 0.2, 0.2, 0.2],
        },
        index=["A", "B", "C", "D", "E"],
    )

    out = compute_inverse_vol_weights(
        features=df,
        portfolio_size=5,
        weight_cap=0.10,
        logger=logger,
    )

    weights = out["weight"]
    assert (weights <= 0.10 + 1e-9).all()
    # Cap is infeasible (0.10 * 5 = 0.50), so sum should be <= 0.50
    assert weights.sum() <= 0.50 + 1e-9


def test_max_position_cap_never_exceeds_cap():
    weights_df = pd.DataFrame({"weight": [0.9, 0.1]}, index=["A", "B"])
    out = apply_max_position_cap(weights_df, max_position_pct=0.20, logger=logging.getLogger("test"))
    assert float(out["weight"].max()) <= 0.20 + 1e-9
    # Infeasible with 2 names under 20% cap -> should leave cash unallocated.
    assert float(out["weight"].sum()) <= 0.40 + 1e-9


def test_correlation_limits_pair_cap_survives_normalization():
    np.random.seed(7)
    idx = pd.date_range("2025-01-01", periods=80, freq="B")
    base = np.cumprod(1.0 + np.random.normal(0, 0.01, len(idx)))
    prices = pd.DataFrame(
        {
            "A": 100 * base,
            "B": 80 * base * (1.0 + np.random.normal(0, 0.001, len(idx))),
            "C": 50 * np.cumprod(1.0 + np.random.normal(0, 0.01, len(idx))),
        },
        index=idx,
    )
    weights_df = pd.DataFrame({"weight": [0.5, 0.4, 0.1]}, index=["A", "B", "C"])
    out = apply_correlation_limits(
        weights_df,
        prices,
        max_corr_weight=0.25,
        corr_threshold=0.7,
        logger=logging.getLogger("test"),
    )
    pair_weight = float(out.loc["A", "weight"] + out.loc["B", "weight"])
    assert pair_weight <= 0.25 + 1e-9

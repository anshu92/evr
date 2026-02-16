import logging
import numpy as np

import pandas as pd

from stock_screener.optimization.risk_parity import (
    SCIPY_AVAILABLE,
    apply_correlation_limits,
    apply_max_position_cap,
    compute_inverse_vol_weights,
    optimize_unified_portfolio,
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


def test_unified_optimizer_enforces_cap_and_corr_constraints():
    np.random.seed(12)
    idx = pd.date_range("2025-01-01", periods=90, freq="B")
    base = np.cumprod(1.0 + np.random.normal(0, 0.008, len(idx)))
    prices = pd.DataFrame(
        {
            "A": 100 * base,
            "B": 90 * base * (1.0 + np.random.normal(0, 0.001, len(idx))),
            "C": 70 * np.cumprod(1.0 + np.random.normal(0, 0.01, len(idx))),
        },
        index=idx,
    )
    features = pd.DataFrame(
        {
            "pred_return": [0.04, 0.03, 0.02],
            "vol_60d_ann": [0.20, 0.21, 0.25],
            "beta": [1.2, 1.1, 0.8],
            "avg_dollar_volume_cad": [12_000_000, 10_000_000, 8_000_000],
        },
        index=["A", "B", "C"],
    )
    base_weights = pd.DataFrame({"weight": [0.45, 0.45, 0.10]}, index=features.index)
    current_weights = pd.Series({"A": 0.30, "B": 0.20, "C": 0.10})
    out = optimize_unified_portfolio(
        base_weights,
        features=features,
        prices=prices,
        current_weights=current_weights,
        alpha_col="pred_return",
        max_position_pct=0.20,
        max_corr_weight=0.25,
        corr_threshold=0.70,
        target_beta=1.0,
        beta_tolerance=0.35,
        risk_penalty=1.0,
        turnover_penalty=1.0,
        cost_penalty=1.0,
        logger=logging.getLogger("test"),
    )
    assert float(out["weight"].max()) <= 0.20 + 1e-8
    assert float(out["weight"].sum()) <= 1.0 + 1e-8
    pair_weight = float(out.loc["A", "weight"] + out.loc["B", "weight"])
    assert pair_weight <= 0.25 + 1e-8


def test_unified_optimizer_turnover_penalty_keeps_weights_closer_to_current():
    if not SCIPY_AVAILABLE:
        return
    np.random.seed(21)
    idx = pd.date_range("2025-01-01", periods=75, freq="B")
    prices = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1.0 + np.random.normal(0, 0.01, len(idx))),
            "B": 80 * np.cumprod(1.0 + np.random.normal(0, 0.01, len(idx))),
            "C": 60 * np.cumprod(1.0 + np.random.normal(0, 0.01, len(idx))),
        },
        index=idx,
    )
    features = pd.DataFrame(
        {
            "pred_return": [0.06, 0.01, 0.00],
            "vol_60d_ann": [0.18, 0.22, 0.25],
            "beta": [1.1, 0.9, 0.8],
            "avg_dollar_volume_cad": [15_000_000, 9_000_000, 7_000_000],
        },
        index=["A", "B", "C"],
    )
    base_weights = pd.DataFrame({"weight": [0.34, 0.33, 0.33]}, index=features.index)
    current_weights = pd.Series({"A": 0.10, "B": 0.45, "C": 0.20})

    low_penalty = optimize_unified_portfolio(
        base_weights,
        features=features,
        prices=prices,
        current_weights=current_weights,
        turnover_penalty=0.0,
        risk_penalty=1.0,
        cost_penalty=0.5,
        max_position_pct=0.25,
        max_corr_weight=0.50,
        logger=logging.getLogger("test"),
    )["weight"]
    high_penalty = optimize_unified_portfolio(
        base_weights,
        features=features,
        prices=prices,
        current_weights=current_weights,
        turnover_penalty=8.0,
        risk_penalty=1.0,
        cost_penalty=0.5,
        max_position_pct=0.25,
        max_corr_weight=0.50,
        logger=logging.getLogger("test"),
    )["weight"]

    low_dist = float(np.abs(low_penalty - current_weights.reindex(low_penalty.index).fillna(0.0)).sum())
    high_dist = float(np.abs(high_penalty - current_weights.reindex(high_penalty.index).fillna(0.0)).sum())
    assert high_dist <= low_dist + 1e-8


def test_unified_optimizer_shrinkage_cov_stable_on_collinear_prices():
    np.random.seed(33)
    idx = pd.date_range("2025-01-01", periods=100, freq="B")
    base = np.cumprod(1.0 + np.random.normal(0, 0.004, len(idx)))
    # Strong collinearity across names makes sample covariance ill-conditioned.
    prices = pd.DataFrame(
        {
            "A": 100.0 * base,
            "B": 90.0 * (base * 1.0002),
            "C": 80.0 * (base * 0.9998),
        },
        index=idx,
    )
    features = pd.DataFrame(
        {
            "pred_return": [0.03, 0.025, 0.02],
            "vol_60d_ann": [0.15, 0.16, 0.17],
            "beta": [1.05, 1.00, 0.95],
            "avg_dollar_volume_cad": [10_000_000, 9_500_000, 9_000_000],
        },
        index=["A", "B", "C"],
    )
    base_weights = pd.DataFrame({"weight": [0.34, 0.33, 0.33]}, index=features.index)

    out = optimize_unified_portfolio(
        base_weights,
        features=features,
        prices=prices,
        current_weights=pd.Series({"A": 0.2, "B": 0.2, "C": 0.2}),
        use_shrinkage_cov=True,
        shrinkage_min_obs=30,
        max_position_pct=0.25,
        max_corr_weight=0.55,
        corr_threshold=0.7,
        logger=logging.getLogger("test"),
    )

    assert np.isfinite(out["weight"]).all()
    assert float(out["weight"].sum()) <= 1.0 + 1e-8
    assert float(out["weight"].max()) <= 0.25 + 1e-8

import logging

import pandas as pd

from stock_screener.pipeline.daily import (
    _apply_instrument_sleeve_constraints,
    _compute_effective_entry_thresholds,
    _lookup_metric_from_sources,
    compute_dynamic_portfolio_size,
)
from stock_screener.config import Config


def test_dynamic_sizing_hard_threshold_log_handles_missing_confidence(caplog):
    logger = logging.getLogger("test_dynamic_sizing")
    logger.setLevel(logging.INFO)

    screened = pd.DataFrame(
        {
            "pred_return": [0.03, 0.02],
            "score": [1.0, 0.9],
        },
        index=["AAA", "BBB"],
    )

    with caplog.at_level(logging.INFO, logger="test_dynamic_sizing"):
        compute_dynamic_portfolio_size(
            screened=screened,
            min_confidence=0.5,
            min_pred_return=0.01,
            max_positions=10,
            model_ic=None,
            logger=logger,
        )

    assert any(
        "2 stocks pass hard thresholds" in rec.getMessage()
        for rec in caplog.records
    )


def test_dynamic_sizing_no_metric_fallback_respects_max_positions():
    logger = logging.getLogger("test_dynamic_sizing")
    logger.setLevel(logging.INFO)

    screened = pd.DataFrame({"foo": [1, 2, 3, 4]}, index=["A", "B", "C", "D"])
    size = compute_dynamic_portfolio_size(
        screened=screened,
        min_confidence=0.5,
        min_pred_return=0.01,
        max_positions=3,
        model_ic=None,
        logger=logger,
    )
    assert size == 3


def test_lookup_metric_from_sources_uses_broader_frames():
    screened = pd.DataFrame(index=["AAA"])
    scored = pd.DataFrame({"pred_return": [0.07], "pred_confidence": [0.82]}, index=["ZZZ"])
    features = pd.DataFrame({"pred_return": [0.03], "pred_confidence": [0.51]}, index=["BBB"])

    assert _lookup_metric_from_sources("ZZZ", "pred_return", [screened, scored, features]) == 0.07
    assert _lookup_metric_from_sources("BBB", "pred_confidence", [screened, scored, features]) == 0.51


def test_dynamic_entry_thresholds_relax_when_prediction_distribution_is_compressed():
    logger = logging.getLogger("test_dynamic_entry_thresholds")
    logger.setLevel(logging.INFO)

    n = 30
    screened = pd.DataFrame(
        {
            "pred_confidence": [0.34 + (i * 0.004) for i in range(n)],
            "pred_return": [0.0015 + (i * 0.0002) for i in range(n)],
        },
        index=[f"T{i}" for i in range(n)],
    )
    cfg = Config(
        entry_min_confidence=0.50,
        entry_min_pred_return=0.01,
        entry_dynamic_thresholds_enabled=True,
        entry_dynamic_min_candidates=20,
        entry_confidence_percentile=0.35,
        entry_min_confidence_floor=0.35,
        entry_pred_return_percentile=0.60,
        entry_min_pred_return_floor=0.0025,
    )

    out = _compute_effective_entry_thresholds(screened, cfg=cfg, logger=logger)

    assert out["dynamic_applied"] is True
    assert 0.35 <= float(out["min_confidence"]) < 0.50
    assert 0.0025 <= float(out["min_pred_return"]) < 0.01


def test_dynamic_entry_thresholds_do_not_apply_when_not_enough_candidates():
    logger = logging.getLogger("test_dynamic_entry_thresholds_small")
    logger.setLevel(logging.INFO)

    screened = pd.DataFrame(
        {
            "pred_confidence": [0.20, 0.21, 0.22],
            "pred_return": [0.001, 0.002, 0.003],
        },
        index=["A", "B", "C"],
    )
    cfg = Config(
        entry_min_confidence=0.50,
        entry_min_pred_return=0.01,
        entry_dynamic_thresholds_enabled=True,
        entry_dynamic_min_candidates=10,
    )

    out = _compute_effective_entry_thresholds(screened, cfg=cfg, logger=logger)

    assert out["dynamic_applied"] is False
    assert float(out["min_confidence"]) == 0.50
    assert float(out["min_pred_return"]) == 0.01


def test_instrument_sleeve_constraints_shift_weight_from_funds_to_equities():
    logger = logging.getLogger("test_instrument_sleeves")
    logger.setLevel(logging.INFO)

    weights = pd.DataFrame({"weight": [0.20, 0.20, 0.20]}, index=["BND", "IEF", "EXAS"])
    screened = pd.DataFrame(
        {
            "quote_type": ["ETF", "ETF", "EQUITY"],
            "sector": [None, None, "Healthcare"],
            "industry": [None, None, "Diagnostics"],
            "log_market_cap": [float("nan"), float("nan"), 10.0],
        },
        index=["BND", "IEF", "EXAS"],
    )
    cfg = Config(
        instrument_sleeve_constraints_enabled=True,
        instrument_fund_max_weight=0.25,
        instrument_equity_min_weight=0.40,
    )

    out, info = _apply_instrument_sleeve_constraints(
        weights,
        screened=screened,
        cfg=cfg,
        logger=logger,
    )

    assert info["applied"] is True
    fund_weight = float(out.loc[["BND", "IEF"], "weight"].sum())
    equity_weight = float(out.loc[["EXAS"], "weight"].sum())
    assert fund_weight <= 0.25 + 1e-9
    assert equity_weight >= 0.40 - 1e-9


def test_instrument_sleeve_constraints_skip_when_only_one_sleeve_present():
    logger = logging.getLogger("test_instrument_sleeves_single")
    logger.setLevel(logging.INFO)

    weights = pd.DataFrame({"weight": [0.30, 0.20]}, index=["BND", "IEF"])
    screened = pd.DataFrame(
        {
            "quote_type": ["ETF", "ETF"],
            "sector": [None, None],
            "industry": [None, None],
            "log_market_cap": [float("nan"), float("nan")],
        },
        index=["BND", "IEF"],
    )
    cfg = Config(
        instrument_sleeve_constraints_enabled=True,
        instrument_fund_max_weight=0.25,
        instrument_equity_min_weight=0.40,
    )

    out, info = _apply_instrument_sleeve_constraints(
        weights,
        screened=screened,
        cfg=cfg,
        logger=logger,
    )

    assert info["applied"] is False
    assert info["reason"] == "single_sleeve_only"
    assert out["weight"].tolist() == [0.30, 0.20]

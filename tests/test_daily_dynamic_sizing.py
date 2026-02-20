import logging

import pandas as pd
import pytest

from stock_screener.pipeline.daily import (
    _apply_instrument_sleeve_constraints,
    _compute_effective_entry_thresholds,
    _compute_ret_per_day_signal,
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


def test_dynamic_entry_thresholds_block_relaxation_when_cross_sectional_spread_is_weak():
    logger = logging.getLogger("test_dynamic_entry_thresholds_spread_guard")
    logger.setLevel(logging.INFO)

    n = 30
    screened = pd.DataFrame(
        {
            "pred_confidence": [0.40 + (i * 0.0005) for i in range(n)],
            "pred_return": [0.0030 + (i * 0.00003) for i in range(n)],
        },
        index=[f"T{i}" for i in range(n)],
    )
    cfg = Config(
        entry_min_confidence=0.50,
        entry_min_pred_return=0.01,
        entry_dynamic_thresholds_enabled=True,
        entry_dynamic_min_candidates=20,
        entry_dynamic_min_conf_top_decile_spread=0.03,
        entry_dynamic_min_pred_top_decile_spread=0.002,
    )

    out = _compute_effective_entry_thresholds(screened, cfg=cfg, logger=logger)

    assert out["dynamic_applied"] is False
    assert "confidence_spread_too_low" in out.get("dynamic_blocked_reasons", [])
    assert "pred_return_spread_too_low" in out.get("dynamic_blocked_reasons", [])
    assert float(out["min_confidence"]) == 0.50
    assert float(out["min_pred_return"]) == 0.01


def test_entry_stress_guard_tightens_thresholds_and_can_recommend_hold_only():
    logger = logging.getLogger("test_entry_stress_guard")
    logger.setLevel(logging.INFO)

    n = 25
    screened = pd.DataFrame(
        {
            "pred_confidence": [0.35 + (i * 0.01) for i in range(n)],
            "pred_return": [0.003 + (i * 0.001) for i in range(n)],
            "market_vol_regime": [2.0 for _ in range(n)],
            "market_breadth": [0.30 for _ in range(n)],
        },
        index=[f"T{i}" for i in range(n)],
    )
    cfg = Config(
        entry_min_confidence=0.50,
        entry_min_pred_return=0.01,
        entry_dynamic_thresholds_enabled=True,
        entry_stress_guard_enabled=True,
        entry_stress_max_vol_stress=0.65,
        entry_stress_min_breadth=0.45,
        entry_stress_confidence_tighten_add=0.05,
        entry_stress_pred_return_tighten_add=0.002,
        entry_stress_hold_only_enabled=True,
    )

    out = _compute_effective_entry_thresholds(screened, cfg=cfg, logger=logger)

    assert out["stress_guard_triggered"] is True
    assert out["dynamic_applied"] is False
    assert "stress_guard_triggered" in out.get("dynamic_blocked_reasons", [])
    assert out["hold_only_recommended"] is True
    assert float(out["min_confidence"]) >= 0.55
    assert float(out["min_pred_return"]) >= 0.012


def test_ret_per_day_signal_clamps_peak_days_and_applies_smoothing():
    logger = logging.getLogger("test_ret_per_day_signal")
    logger.setLevel(logging.INFO)

    features = pd.DataFrame(
        {
            "pred_return": [0.10, 0.10, 0.10],
            "pred_peak_days": [0.2, 5.0, 100.0],
        },
        index=["A", "B", "C"],
    )
    cfg = Config(
        ret_per_day_min_peak_days=1.0,
        ret_per_day_max_peak_days=10.0,
        ret_per_day_smoothing_k=1.0,
        ret_per_day_shift_alert_enabled=False,
    )

    info = _compute_ret_per_day_signal(features, cfg=cfg, logger=logger, previous_run_meta=None)

    assert info["enabled"] is True
    assert float(features.loc["A", "pred_peak_days"]) == 1.0
    assert float(features.loc["C", "pred_peak_days"]) == 10.0
    assert float(features.loc["A", "ret_per_day"]) == pytest.approx(0.10 / (1.0 + 1.0), abs=1e-12)
    assert float(features.loc["C", "ret_per_day"]) == pytest.approx(0.10 / (10.0 + 1.0), abs=1e-12)
    assert info["peak_days_clip_low_count"] == 1
    assert info["peak_days_clip_high_count"] == 1


def test_ret_per_day_signal_emits_shift_alert_on_distribution_change():
    logger = logging.getLogger("test_ret_per_day_shift")
    logger.setLevel(logging.INFO)

    features = pd.DataFrame(
        {
            "pred_return": [0.03 + i * 0.005 for i in range(40)],
            "pred_peak_days": [1.0 for _ in range(40)],
        },
        index=[f"T{i}" for i in range(40)],
    )
    previous_run_meta = {
        "ret_per_day_signal": {
            "ret_per_day_stats": {
                "n": 40,
                "mean": 0.01,
                "p90": 0.015,
                "std": 0.004,
            }
        }
    }
    cfg = Config(
        ret_per_day_min_peak_days=1.0,
        ret_per_day_max_peak_days=10.0,
        ret_per_day_smoothing_k=1.0,
        ret_per_day_shift_alert_enabled=True,
        ret_per_day_shift_alert_min_samples=20,
        ret_per_day_mean_shift_alert_pct=0.20,
        ret_per_day_p90_shift_alert_pct=0.20,
        ret_per_day_std_shift_alert_pct=0.20,
    )

    info = _compute_ret_per_day_signal(features, cfg=cfg, logger=logger, previous_run_meta=previous_run_meta)

    assert info["enabled"] is True
    assert info["shift_alert_triggered"] is True
    assert len(info.get("shift_alerts", [])) >= 1


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

import logging

import pandas as pd

from stock_screener.pipeline.daily import (
    _lookup_metric_from_sources,
    compute_dynamic_portfolio_size,
)


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

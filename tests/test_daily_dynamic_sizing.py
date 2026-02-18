import logging

import pandas as pd

from stock_screener.pipeline.daily import compute_dynamic_portfolio_size


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

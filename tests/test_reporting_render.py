import logging

import pandas as pd

from stock_screener.reporting.render import render_reports


def test_render_reports_preserves_zero_action_fields(tmp_path):
    logger = logging.getLogger("test")
    reports_dir = tmp_path / "reports"

    screened = pd.DataFrame(
        {
            "score": [1.0],
            "last_close_cad": [100.0],
            "ret_60d": [0.01],
            "ret_120d": [0.02],
            "vol_60d_ann": [0.20],
            "avg_dollar_volume_cad": [5_000_000.0],
            "rsi_14": [55.0],
        },
        index=["AAPL"],
    )
    weights = pd.DataFrame(
        {
            "weight": [1.0],
            "score": [1.0],
            "last_close_cad": [100.0],
            "ret_60d": [0.01],
            "vol_60d_ann": [0.20],
            "avg_dollar_volume_cad": [5_000_000.0],
            "shares": [1.0],
        },
        index=["AAPL"],
    )
    trade_actions = [
        {
            "ticker": "AAPL",
            "action": "SELL",
            "reason": "TEST",
            "shares": 1.0,
            "price_cad": 100.0,
            "days_held": 0,
            "entry_price": 100.0,
            "realized_gain_pct": 0.0,
        }
    ]

    render_reports(
        reports_dir=reports_dir,
        run_meta={},
        universe_meta={"us": {}, "tsx": {}, "total_requested": 1},
        screened=screened,
        weights=weights,
        trade_actions=trade_actions,
        logger=logger,
    )

    txt = (reports_dir / "daily_report.txt").read_text(encoding="utf-8")
    assert "days_held=0" in txt
    assert "gain=+0.00%" in txt

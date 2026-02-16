import pandas as pd
import pytest

from stock_screener.modeling.eval import evaluate_topn_returns, summarize_topn_returns


def test_summarize_topn_returns_includes_per_day_metrics():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "mean_ret": [0.05, 0.03],
            "net_ret": [0.049, 0.029],
        }
    )
    out = summarize_topn_returns(daily, holding_days=5)
    assert out["mean_ret"] == pytest.approx(0.04)
    assert out["mean_net_ret"] == pytest.approx(0.039)
    assert out["mean_ret_per_day"] == pytest.approx(0.008)
    assert out["mean_net_ret_per_day"] == pytest.approx(0.0078)
    assert out["holding_days"] == 5


def test_evaluate_topn_returns_passes_holding_days_to_summary():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]),
            "future_ret": [0.02, 0.05, 0.01, 0.03],
            "pred": [0.2, 0.9, 0.1, 0.8],
        }
    )
    out = evaluate_topn_returns(
        df,
        date_col="date",
        label_col="future_ret",
        pred_col="pred",
        top_n=1,
        cost_bps=0.0,
        holding_days=5,
    )
    summary = out["summary"]
    assert summary["holding_days"] == 5
    assert summary["mean_ret"] == pytest.approx(0.04)
    assert summary["mean_ret_per_day"] == pytest.approx(0.008)

import pandas as pd
import pytest

from stock_screener.modeling.eval import (
    aggregate_walk_forward_results,
    evaluate_model_promotion_gates,
    simulate_realistic_portfolio,
)


def test_model_promotion_gates_pass_when_all_thresholds_met():
    realistic = {
        "return_per_day": 0.00035,
        "cost_adjusted_sharpe": 0.92,
        "max_drawdown": -0.12,
        "turnover_efficiency": 0.45,
        "avg_turnover": 0.35,
    }
    walk_forward = {"consistency": 0.75, "n_periods": 3}

    out = evaluate_model_promotion_gates(
        realistic_metrics=realistic,
        walk_forward_results=walk_forward,
        thresholds={
            "min_return_per_day": 0.0002,
            "min_cost_adjusted_sharpe": 0.5,
            "max_drawdown": -0.25,
            "min_consistency": 0.55,
            "min_turnover_efficiency": 0.20,
            "max_avg_turnover": 0.80,
            "min_periods": 2,
        },
    )
    assert out["passed"] is True
    assert all(g["passed"] for g in out["gates"])


def test_model_promotion_gates_fail_on_drawdown_and_consistency():
    realistic = {
        "return_per_day": 0.00030,
        "cost_adjusted_sharpe": 0.70,
        "max_drawdown": -0.40,
        "turnover_efficiency": 0.35,
        "avg_turnover": 0.45,
    }
    walk_forward = {"consistency": 0.20, "n_periods": 3}
    out = evaluate_model_promotion_gates(realistic_metrics=realistic, walk_forward_results=walk_forward)

    assert out["passed"] is False
    failed = {g["name"] for g in out["gates"] if not g["passed"]}
    assert "max_drawdown" in failed
    assert "consistency_positive_sharpe_periods" in failed


def test_model_promotion_gates_use_walk_forward_aggregate_when_available():
    realistic = {
        "return_per_day": -0.0010,
        "cost_adjusted_sharpe": -1.0,
        "max_drawdown": -0.60,
        "turnover_efficiency": -0.1,
        "avg_turnover": 0.95,
    }
    walk_forward = {
        "consistency": 0.67,
        "n_periods": 3,
        "aggregate": {
            "return_per_day": {"mean": 0.0004},
            "cost_adjusted_sharpe": {"mean": 0.8},
            "max_drawdown": {"mean": -0.20},
            "turnover_efficiency": {"mean": 0.35},
            "avg_turnover": {"mean": 0.55},
        },
    }
    out = evaluate_model_promotion_gates(realistic_metrics=realistic, walk_forward_results=walk_forward)
    assert out["passed"] is True


def test_model_promotion_gates_optional_calibration_gates():
    realistic = {
        "return_per_day": 0.00035,
        "cost_adjusted_sharpe": 0.92,
        "max_drawdown": -0.12,
        "turnover_efficiency": 0.45,
        "avg_turnover": 0.35,
    }
    walk_forward = {"consistency": 0.75, "n_periods": 3}

    pass_out = evaluate_model_promotion_gates(
        realistic_metrics=realistic,
        walk_forward_results=walk_forward,
        calibration_metrics={"calibration_error": 0.004, "calibration_slope": 0.80},
        thresholds={"max_calibration_error": 0.01, "min_calibration_slope": 0.25},
    )
    assert pass_out["passed"] is True
    assert all(g["passed"] for g in pass_out["gates"])

    fail_out = evaluate_model_promotion_gates(
        realistic_metrics=realistic,
        walk_forward_results=walk_forward,
        calibration_metrics={"calibration_error": 0.020, "calibration_slope": 0.10},
        thresholds={"max_calibration_error": 0.01, "min_calibration_slope": 0.25},
    )
    assert fail_out["passed"] is False
    failed = {g["name"] for g in fail_out["gates"] if not g["passed"]}
    assert "calibration_error_cap" in failed
    assert "calibration_slope_floor" in failed


def test_walk_forward_aggregate_emits_pbo_proxy():
    period_results = [
        {"sharpe_ratio": 1.1, "return_per_day": 0.0006},
        {"sharpe_ratio": 0.2, "return_per_day": 0.0001},
        {"sharpe_ratio": -0.5, "return_per_day": -0.0002},
    ]
    out = aggregate_walk_forward_results(period_results)
    assert out["n_periods"] == 3
    assert "pbo_proxy" in out
    assert 0.0 <= float(out["pbo_proxy"]) <= 1.0


def test_model_promotion_gates_optional_pbo_proxy_gate():
    realistic = {
        "return_per_day": 0.00035,
        "cost_adjusted_sharpe": 0.92,
        "max_drawdown": -0.12,
        "turnover_efficiency": 0.45,
        "avg_turnover": 0.35,
    }
    walk_forward = {"consistency": 0.75, "n_periods": 3, "pbo_proxy": 0.60}
    out = evaluate_model_promotion_gates(
        realistic_metrics=realistic,
        walk_forward_results=walk_forward,
        thresholds={"max_pbo_proxy": 0.45},
    )
    assert out["passed"] is False
    failed = {g["name"] for g in out["gates"] if not g["passed"]}
    assert "pbo_proxy_cap" in failed


def test_walk_forward_pbo_proxy_penalizes_unstable_paths():
    robust = aggregate_walk_forward_results(
        [
            {"sharpe_ratio": 1.2, "return_per_day": 0.00060, "max_drawdown": -0.10},
            {"sharpe_ratio": 1.1, "return_per_day": 0.00055, "max_drawdown": -0.11},
            {"sharpe_ratio": 0.9, "return_per_day": 0.00050, "max_drawdown": -0.12},
        ]
    )
    unstable = aggregate_walk_forward_results(
        [
            {"sharpe_ratio": 1.0, "return_per_day": 0.00100, "max_drawdown": -0.10},
            {"sharpe_ratio": -0.5, "return_per_day": -0.00150, "max_drawdown": -0.40},
            {"sharpe_ratio": 0.1, "return_per_day": 0.00020, "max_drawdown": -0.35},
        ]
    )
    assert float(robust["pbo_proxy"]) < float(unstable["pbo_proxy"])


def test_simulate_realistic_portfolio_uses_entry_horizon_return():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])
    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["A"] * 4 + ["B"] * 4,
            "future_ret": [
                0.21,   # rebalance day 1, selected
                -0.90,  # should NOT impact day 2 when holding from day 1
                0.21,   # rebalance day 3, selected
                -0.90,  # should NOT impact day 4 when holding from day 3
                -0.10,
                0.50,
                -0.10,
                0.50,
            ],
            "pred": [
                10.0, 0.0, 10.0, 0.0,  # ticker A selected only on rebalance dates
                1.0, 0.0, 1.0, 0.0,
            ],
        }
    )
    out = simulate_realistic_portfolio(
        df,
        date_col="date",
        ticker_col="ticker",
        label_col="future_ret",
        pred_col="pred",
        top_n=1,
        hold_days=2,
        cost_bps=0.0,
    )
    daily = out["daily"]
    assert len(daily) == 4
    expected_daily = (1.21 ** 0.5) - 1.0
    assert daily["return"].tolist() == pytest.approx([expected_daily] * 4, abs=1e-9)


def test_simulate_realistic_portfolio_hysteresis_reduces_turnover():
    dates = pd.to_datetime(
        [
            "2026-01-01", "2026-01-02",
            "2026-01-03", "2026-01-04",
            "2026-01-05", "2026-01-06",
        ]
    )
    rows = []
    for d in dates:
        if d in {pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")}:
            ranks = [("A", 1.0), ("B", 0.9), ("C", 0.8), ("D", 0.7)]
        else:
            ranks = [("C", 1.0), ("D", 0.9), ("A", 0.8), ("B", 0.7)]
        for t, p in ranks:
            rows.append({"date": d, "ticker": t, "future_ret": 0.05, "pred": p})
    df = pd.DataFrame(rows)

    no_hyst = simulate_realistic_portfolio(
        df,
        date_col="date",
        ticker_col="ticker",
        label_col="future_ret",
        pred_col="pred",
        top_n=2,
        hold_days=2,
        cost_bps=0.0,
        rebalance_hysteresis=0.0,
    )["summary"]["avg_turnover"]
    with_hyst = simulate_realistic_portfolio(
        df,
        date_col="date",
        ticker_col="ticker",
        label_col="future_ret",
        pred_col="pred",
        top_n=2,
        hold_days=2,
        cost_bps=0.0,
        rebalance_hysteresis=1.0,
    )["summary"]["avg_turnover"]

    assert float(with_hyst) < float(no_hyst)

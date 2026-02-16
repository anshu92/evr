from stock_screener.modeling.eval import evaluate_model_promotion_gates


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

import numpy as np
import pandas as pd

from stock_screener.modeling.model import compute_regime_gate_weights, predict_regime_gated


def test_regime_gate_weights_follow_market_signals():
    features = pd.DataFrame(
        {
            "market_trend_20d": [0.08, -0.07, 0.00],
            "market_vol_regime": [0.8, 1.7, 1.0],
            "market_breadth": [0.75, 0.35, 0.50],
        },
        index=["bull_case", "bear_case", "neutral_case"],
    )

    gates = compute_regime_gate_weights(features)
    assert list(gates.columns) == ["bull", "neutral", "bear"]
    np.testing.assert_allclose(gates.sum(axis=1).values, np.ones(len(gates)), rtol=1e-8)

    assert gates.loc["bull_case", "bull"] > 0.5
    assert gates.loc["bear_case", "bear"] > 0.5
    assert gates.loc["neutral_case", "neutral"] == gates.loc["neutral_case"].max()


def test_predict_regime_gated_blends_experts_and_base():
    idx = pd.Index(["x", "y"])
    base = pd.Series([0.10, 0.10], index=idx)
    gates = pd.DataFrame(
        {
            "bull": [1.0, 0.0],
            "neutral": [0.0, 0.0],
            "bear": [0.0, 1.0],
        },
        index=idx,
    )
    regime_preds = {
        "bull": pd.Series([0.30, 0.30], index=idx),
        "bear": pd.Series([-0.20, -0.20], index=idx),
        "neutral": pd.Series([0.0, 0.0], index=idx),
    }

    out = predict_regime_gated(base, regime_preds=regime_preds, gate_weights=gates, base_blend=0.2)
    assert np.isclose(float(out.loc["x", "pred_return"]), 0.26, atol=1e-8)
    assert np.isclose(float(out.loc["y", "pred_return"]), -0.14, atol=1e-8)
    assert out.loc["x", "regime"] == "bull"
    assert out.loc["y", "regime"] == "bear"

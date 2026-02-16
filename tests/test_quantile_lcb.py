from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stock_screener.config import Config
from stock_screener.modeling.model import predict_quantile_lcb, save_ensemble


class _FakeModel:
    def __init__(self, preds: list[float]) -> None:
        self._preds = np.asarray(preds, dtype=float)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self._preds


def test_predict_quantile_lcb_enforces_monotonic_quantiles_and_lcb():
    features = pd.DataFrame({"x": [1.0, 2.0]}, index=["A", "B"])
    quantile_ensembles = {
        # Intentionally non-monotonic for row A to verify sorting guard.
        "q10": ([_FakeModel([0.03, 0.00])], None),
        "q50": ([_FakeModel([0.02, 0.01])], None),
        "q90": ([_FakeModel([0.01, 0.05])], None),
    }
    out = predict_quantile_lcb(
        quantile_ensembles,
        features,
        feature_cols=["x"],
        lcb_risk_aversion=0.5,
    )
    assert (out["pred_return_q10"] <= out["pred_return_q50"]).all()
    assert (out["pred_return_q50"] <= out["pred_return_q90"]).all()

    expected_spread = out["pred_return_q90"] - out["pred_return_q10"]
    expected_lcb = out["pred_return_q50"] - 0.5 * expected_spread
    assert np.allclose(out["pred_quantile_spread"], expected_spread)
    assert np.allclose(out["pred_return_lcb"], expected_lcb)


def test_save_ensemble_persists_quantile_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    save_ensemble(
        manifest,
        model_rel_paths=["xgb_model_0.json"],
        model_types=["xgboost"],
        weights=[1.0],
        quantile_models={
            "q10": {"models": ["quantile_q10.txt"], "model_types": ["lightgbm"], "weights": None},
            "q50": {"models": ["quantile_q50.txt"], "model_types": ["lightgbm"], "weights": None},
            "q90": {"models": ["quantile_q90.txt"], "model_types": ["lightgbm"], "weights": None},
        },
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "quantile_models" in payload
    assert sorted(payload["quantile_models"].keys()) == ["q10", "q50", "q90"]


def test_config_parses_quantile_env(monkeypatch):
    monkeypatch.setenv("QUANTILE_MODELS_ENABLED", "1")
    monkeypatch.setenv("LCB_RISK_AVERSION", "0.75")
    cfg = Config.from_env()
    assert cfg.quantile_models_enabled is True
    assert cfg.lcb_risk_aversion == 0.75

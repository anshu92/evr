from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
try:
    import xgboost as xgb
except Exception as _xgb_err:  # pragma: no cover
    xgb = None
    _XGB_IMPORT_ERROR = _xgb_err


FEATURE_COLUMNS = [
    "last_close_cad",
    "avg_dollar_volume_cad",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "vol_20d_ann",
    "vol_60d_ann",
    "rsi_14",
    "ma20_ratio",
    "ma50_ratio",
    "ma200_ratio",
    "is_tsx",
]


def _require_xgb() -> None:
    if xgb is None:  # pragma: no cover
        raise RuntimeError(
            "XGBoost is not available in this environment. "
            "On macOS you typically need the OpenMP runtime (libomp). "
            "GitHub Actions ubuntu runners will work out of the box. "
            f"Original error: {_XGB_IMPORT_ERROR}"
        )


def build_model(random_state: int = 42):
    """Build a strong baseline boosted-tree model for next-horizon returns."""

    _require_xgb()
    return xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=5,
        objective="reg:squarederror",
        n_jobs=0,
        random_state=random_state,
    )


def _coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in FEATURE_COLUMNS:
        if c not in x.columns:
            x[c] = np.nan
    # Convert bool to int for sklearn
    if "is_tsx" in x.columns:
        x["is_tsx"] = x["is_tsx"].astype(int, errors="ignore")
    return x[FEATURE_COLUMNS]


def predict(model, features: pd.DataFrame) -> pd.Series:
    _require_xgb()
    x = _coerce_features(features)
    if isinstance(model, xgb.Booster):
        preds = model.predict(xgb.DMatrix(x))
    else:
        preds = model.predict(x)
    return pd.Series(preds, index=features.index, name="pred_return")


def predict_score(model, features: pd.DataFrame) -> pd.Series:
    _require_xgb()
    x = _coerce_features(features)
    if isinstance(model, xgb.Booster):
        preds = model.predict(xgb.DMatrix(x))
    else:
        preds = model.predict(x)
    return pd.Series(preds, index=features.index, name="pred_score")


def predict_ensemble(models: list, weights: list[float] | None, features: pd.DataFrame) -> pd.Series:
    _require_xgb()
    if not models:
        raise ValueError("No models provided for ensemble prediction")
    if weights is None:
        w = np.ones(len(models), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(models):
            raise ValueError("weights length must match models length")
    w = w / float(w.sum())

    out = np.zeros(len(features), dtype=float)
    for i, m in enumerate(models):
        out += w[i] * predict(m, features).astype(float).values
    return pd.Series(out, index=features.index, name="pred_return")


def save_model(model, path: str | Path) -> None:
    _require_xgb()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Save as JSON (non-pickle, portable).
    # XGBoost's sklearn wrapper can raise `_estimator_type` errors in some envs when calling save_model().
    # Saving the underlying Booster is stable across environments (including GitHub Actions).
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    booster.save_model(str(p))


def save_ensemble(manifest_path: str | Path, model_rel_paths: list[str], weights: list[float] | None = None) -> None:
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "xgboost_ensemble_v1",
        "models": model_rel_paths,
        "weights": weights,
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_bundle(
    manifest_path: str | Path,
    *,
    ranker_rel_path: str | None,
    regressor_rel_paths: list[str],
    regressor_weights: list[float] | None = None,
    metadata_rel_path: str | None = None,
) -> None:
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "xgboost_dual_v1",
        "ranker": ranker_rel_path,
        "regressor": {"models": regressor_rel_paths, "weights": regressor_weights},
        "metadata": metadata_rel_path,
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_model(path: str | Path):
    # Security: Avoid pickle/joblib. XGBoost JSON is safe to load.
    _require_xgb()
    booster = xgb.Booster()
    booster.load_model(str(Path(path)))
    return booster


def load_ensemble(manifest_path: str | Path) -> tuple[list, list[float] | None]:
    _require_xgb()
    mp = Path(manifest_path)
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    if manifest.get("type") != "xgboost_ensemble_v1":
        raise ValueError("Unsupported ensemble manifest type")
    model_rel = manifest.get("models") or []
    weights = manifest.get("weights")
    base = mp.parent
    models = [load_model(base / rel) for rel in model_rel]
    return models, weights




def load_bundle(manifest_path: str | Path) -> dict[str, object]:
    _require_xgb()
    mp = Path(manifest_path)
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    base = mp.parent
    mtype = manifest.get("type")

    if mtype == "xgboost_ensemble_v1":
        models, weights = load_ensemble(mp)
        return {"type": mtype, "ranker": None, "regressor_models": models, "regressor_weights": weights, "metadata": None}

    if mtype != "xgboost_dual_v1":
        raise ValueError("Unsupported ensemble manifest type")

    ranker_rel = manifest.get("ranker")
    reg_payload = manifest.get("regressor") or {}
    reg_models = reg_payload.get("models") or []
    reg_weights = reg_payload.get("weights")
    metadata = manifest.get("metadata")

    ranker = load_model(base / ranker_rel) if ranker_rel else None
    regressor_models = [load_model(base / rel) for rel in reg_models]
    return {
        "type": mtype,
        "ranker": ranker,
        "regressor_models": regressor_models,
        "regressor_weights": reg_weights,
        "metadata": metadata,
    }

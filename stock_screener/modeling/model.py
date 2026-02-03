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

try:
    import lightgbm as lgb
except Exception as _lgb_err:  # pragma: no cover
    lgb = None
    _LGB_IMPORT_ERROR = _lgb_err


FEATURE_COLUMNS = [
    "last_close_cad",
    "avg_dollar_volume_cad",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "ret_accel_20_120",
    "vol_20d_ann",
    "vol_60d_ann",
    "vol_ratio_20_60",
    "rsi_14",
    "ma20_ratio",
    "ma50_ratio",
    "ma200_ratio",
    "drawdown_60d",
    "dist_52w_high",
    "dist_52w_low",
    "vol_anom_30d",
    "rank_ret_20d",
    "rank_ret_60d",
    "rank_vol_60d",
    "rank_avg_dollar_volume",
    "fx_ret_5d",
    "fx_ret_20d",
    "is_tsx",
    # Fundamental features
    "log_market_cap",
    "beta",
    # Fundamental composite scores
    "value_score",
    "quality_score",
    "growth_score",
    "pe_discount",
    "roc_growth",
    "value_momentum",
    "vol_size",
    "quality_growth",
]

# Technical features only (no fundamentals) - use for training to avoid lookahead bias
TECHNICAL_FEATURES_ONLY = [
    "last_close_cad",
    "avg_dollar_volume_cad",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "ret_accel_20_120",
    "vol_20d_ann",
    "vol_60d_ann",
    "vol_ratio_20_60",
    "rsi_14",
    "ma20_ratio",
    "ma50_ratio",
    "ma200_ratio",
    "drawdown_60d",
    "dist_52w_high",
    "dist_52w_low",
    "vol_anom_30d",
    "rank_ret_20d",
    "rank_ret_60d",
    "rank_vol_60d",
    "rank_avg_dollar_volume",
    "fx_ret_5d",
    "fx_ret_20d",
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


def _require_lgb() -> None:
    if lgb is None:  # pragma: no cover
        raise RuntimeError(
            "LightGBM is not available in this environment. "
            f"Original error: {_LGB_IMPORT_ERROR}"
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


def build_ranker(random_state: int = 42):
    """Build a baseline learning-to-rank model for cross-sectional ordering."""

    _require_xgb()
    return xgb.XGBRanker(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=5,
        objective="rank:pairwise",
        eval_metric="ndcg",
        n_jobs=0,
        random_state=random_state,
    )


def build_lgbm_model(random_state: int = 42):
    """Build a LightGBM regressor for ensemble diversity."""
    
    _require_lgb()
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=random_state,
        verbose=-1,
    )


def _coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    missing = []
    for c in FEATURE_COLUMNS:
        if c not in x.columns:
            x[c] = np.nan
            missing.append(c)
    
    # Log warning if features are missing
    if missing:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Missing %d feature columns (filled with NaN): %s", len(missing), missing[:10])
    
    # Convert bool to int for sklearn
    if "is_tsx" in x.columns:
        x["is_tsx"] = x["is_tsx"].astype(int, errors="ignore")
    return x[FEATURE_COLUMNS]


def predict(model, features: pd.DataFrame) -> pd.Series:
    x = _coerce_features(features)
    
    # Handle different model types
    if xgb and isinstance(model, xgb.Booster):
        _require_xgb()
        preds = model.predict(xgb.DMatrix(x))
    elif lgb and isinstance(model, lgb.Booster):
        _require_lgb()
        preds = model.predict(x)
    elif lgb and hasattr(model, '__class__') and 'LGB' in model.__class__.__name__:
        # LightGBM sklearn wrapper
        _require_lgb()
        preds = model.predict(x)
    else:
        # XGBoost sklearn wrapper or other
        _require_xgb()
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


def predict_ensemble_with_uncertainty(models: list, weights: list[float] | None, features: pd.DataFrame) -> pd.DataFrame:
    """Return mean prediction, uncertainty (std across models), and confidence."""
    if not models:
        raise ValueError("No models provided for ensemble prediction")
    
    # Get predictions from all models
    preds_list = [predict(m, features).values for m in models]
    preds_array = np.array(preds_list)  # shape: (n_models, n_samples)
    
    # Apply weights if provided
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(models):
            raise ValueError("weights length must match models length")
        w = w / float(w.sum())
        # Weighted mean
        pred_mean = np.average(preds_array, axis=0, weights=w)
    else:
        pred_mean = np.mean(preds_array, axis=0)
    
    # Uncertainty as standard deviation across models (model disagreement)
    pred_std = np.std(preds_array, axis=0)
    
    # Confidence: higher when models agree (lower std)
    # Scale to 0-1 range where 1 = perfect agreement
    pred_confidence = 1.0 / (1.0 + pred_std)
    
    return pd.DataFrame({
        "pred_return": pred_mean,
        "pred_uncertainty": pred_std,
        "pred_confidence": pred_confidence,
    }, index=features.index)


def save_model(model, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle LightGBM models
    if lgb and hasattr(model, '__class__') and 'LGB' in model.__class__.__name__:
        _require_lgb()
        model.booster_.save_model(str(p))
    elif lgb and isinstance(model, lgb.Booster):
        _require_lgb()
        model.save_model(str(p))
    else:
        # XGBoost models
        _require_xgb()
        # Save as JSON (non-pickle, portable).
        # XGBoost's sklearn wrapper can raise `_estimator_type` errors in some envs when calling save_model().
        # Saving the underlying Booster is stable across environments (including GitHub Actions).
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        booster.save_model(str(p))


def save_ensemble(manifest_path: str | Path, model_rel_paths: list[str], model_types: list[str] | None = None, weights: list[float] | None = None) -> None:
    """Save ensemble manifest with model paths and types."""
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "mixed_ensemble_v1",  # Support both XGBoost and LightGBM
        "models": model_rel_paths,
        "model_types": model_types or ["xgboost"] * len(model_rel_paths),
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


def load_model(path: str | Path, model_type: str = "xgboost"):
    """Load a model from disk. Supports both XGBoost and LightGBM."""
    p = Path(path)
    
    if model_type == "lightgbm":
        _require_lgb()
        return lgb.Booster(model_file=str(p))
    else:
        # Default to XGBoost for backward compatibility
        _require_xgb()
        booster = xgb.Booster()
        booster.load_model(str(p))
        return booster


def load_ensemble(manifest_path: str | Path) -> tuple[list, list[float] | None]:
    mp = Path(manifest_path)
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    manifest_type = manifest.get("type")
    
    # Support both old and new formats
    if manifest_type == "xgboost_ensemble_v1":
        _require_xgb()
        model_rel = manifest.get("models") or []
        weights = manifest.get("weights")
        base = mp.parent
        models = [load_model(base / rel, "xgboost") for rel in model_rel]
        return models, weights
    elif manifest_type == "mixed_ensemble_v1":
        model_rel = manifest.get("models") or []
        model_types = manifest.get("model_types") or ["xgboost"] * len(model_rel)
        weights = manifest.get("weights")
        base = mp.parent
        models = [load_model(base / rel, mtype) for rel, mtype in zip(model_rel, model_types)]
        return models, weights
    else:
        raise ValueError(f"Unsupported ensemble manifest type: {manifest_type}")


def load_bundle(manifest_path: str | Path) -> dict[str, object]:
    mp = Path(manifest_path)
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    base = mp.parent
    mtype = manifest.get("type")

    if mtype == "xgboost_ensemble_v1":
        models, weights = load_ensemble(mp)
        return {"type": mtype, "ranker": None, "regressor_models": models, "regressor_weights": weights, "metadata": None}
    
    if mtype == "mixed_ensemble_v1":
        models, weights = load_ensemble(mp)
        return {"type": mtype, "ranker": None, "regressor_models": models, "regressor_weights": weights, "metadata": None}

    if mtype != "xgboost_dual_v1":
        raise ValueError(f"Unsupported ensemble manifest type: {mtype}")
    ranker_rel = manifest.get("ranker")
    reg_payload = manifest.get("regressor") or {}
    reg_models = reg_payload.get("models") or []
    reg_weights = reg_payload.get("weights")
    metadata = manifest.get("metadata")

    ranker = load_model(base / ranker_rel, "xgboost") if ranker_rel else None
    regressor_models = [load_model(base / rel, "xgboost") for rel in reg_models]
    return {
        "type": mtype,
        "ranker": ranker,
        "regressor_models": regressor_models,
        "regressor_weights": reg_weights,
        "metadata": metadata,
    }

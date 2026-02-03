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
    # Momentum quality features
    "momentum_reversal",
    "momentum_acceleration",
    "ret_20d_lagged",
    "ret_60d_lagged",
    # HIGH-IMPACT: Volatility-adjusted returns (Sharpe-like signals)
    "ret_5d_sharpe",
    "ret_20d_sharpe",
    "ret_60d_sharpe",
    # HIGH-IMPACT: Volume-price divergence signals
    "volume_momentum_20d",
    "volume_surge_5d",
    "price_volume_div",
    # HIGH-IMPACT: Mean reversion signals
    "ma20_zscore",
    "mean_reversion_signal",
    # HIGH-IMPACT: Trend quality features
    "ret_consistency_20d",
    "up_days_ratio_20d",
    # Relative momentum (vs market)
    "relative_momentum_20d",
    "relative_momentum_60d",
    # Cross-sectional ranks (position relative to peers on each date)
    "rank_ret_5d",
    "rank_ret_20d",
    "rank_ret_60d",
    "rank_vol_60d",
    "rank_avg_dollar_volume",
    "rank_ret_5d_sharpe",
    "rank_momentum_reversal",
    "rank_ma20_zscore",
    "momentum_strength",
    "fx_ret_5d",
    "fx_ret_20d",
    "is_tsx",
    # Fundamental features
    "log_market_cap",
    "beta",
    # Target-encoded categorical features (meaningful sector/industry signals)
    "sector_target_enc",
    "industry_target_enc",
    # Market regime features (same for all stocks on a given day)
    "market_vol_regime",  # Current market volatility vs historical
    "market_trend_20d",   # Market 20-day return
    "market_breadth",     # % of stocks above their 20-day MA
    "market_momentum_accel",  # Market momentum acceleration
    # Macro regime indicators
    "vix",  # CBOE Volatility Index
    "treasury_10y",  # 10-Year Treasury Yield
    "treasury_13w",  # 13-Week Treasury Bill
    "yield_curve_slope",  # 10Y - 3M yield spread
    # HIGH-IMPACT: Feature interaction terms
    "sharpe_x_rank",  # Sharpe ratio × momentum rank
    "momentum_vol_interaction",  # Momentum × volatility
    "rsi_momentum_interaction",  # RSI extreme × momentum direction
    "size_momentum_interaction",  # Size × relative momentum
    "zscore_reversal",  # Mean reversion potential
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
    # Momentum quality features
    "momentum_reversal",
    "momentum_acceleration",
    "ret_20d_lagged",
    "ret_60d_lagged",
    # HIGH-IMPACT: Volatility-adjusted returns (Sharpe-like signals)
    "ret_5d_sharpe",
    "ret_20d_sharpe",
    "ret_60d_sharpe",
    # HIGH-IMPACT: Volume-price divergence signals
    "volume_momentum_20d",
    "volume_surge_5d",
    "price_volume_div",
    # HIGH-IMPACT: Mean reversion signals
    "ma20_zscore",
    "mean_reversion_signal",
    # HIGH-IMPACT: Trend quality features
    "ret_consistency_20d",
    "up_days_ratio_20d",
    # Relative momentum (vs market)
    "relative_momentum_20d",
    "relative_momentum_60d",
    # Cross-sectional ranks
    "rank_ret_5d",
    "rank_ret_20d",
    "rank_ret_60d",
    "rank_vol_60d",
    "rank_avg_dollar_volume",
    "rank_ret_5d_sharpe",
    "rank_momentum_reversal",
    "rank_ma20_zscore",
    "momentum_strength",
    "fx_ret_5d",
    "fx_ret_20d",
    "is_tsx",
    # Market regime features (same for all stocks on a given day)
    "market_vol_regime",
    "market_trend_20d",
    "market_breadth",
    "market_momentum_accel",
    # Macro regime indicators
    "vix",
    "treasury_10y",
    "treasury_13w",
    "yield_curve_slope",
    # HIGH-IMPACT: Feature interaction terms
    "sharpe_x_rank",
    "momentum_vol_interaction",
    "rsi_momentum_interaction",
    "size_momentum_interaction",
    "zscore_reversal",
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
    """Build a regularized boosted-tree model for next-horizon returns."""

    _require_xgb()
    return xgb.XGBRegressor(
        n_estimators=400,  # Balanced: quality vs speed
        learning_rate=0.025,
        max_depth=5,  # Moderate depth
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=3.0,  # L2 regularization
        reg_alpha=0.3,  # L1 regularization
        min_child_weight=8,
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
    """Build a regularized LightGBM regressor for ensemble diversity."""
    
    _require_lgb()
    return lgb.LGBMRegressor(
        n_estimators=350,  # Balanced: quality vs speed
        learning_rate=0.025,
        max_depth=5,  # Moderate depth
        num_leaves=20,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=3.0,  # L2 regularization
        reg_alpha=0.3,  # L1 regularization
        min_child_samples=30,
        random_state=random_state,
        verbose=-1,
    )


def _coerce_features(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Prepare features for model prediction.
    
    Args:
        df: Raw features DataFrame
        feature_cols: Feature columns to use. If None, uses FEATURE_COLUMNS.
    """
    cols = feature_cols if feature_cols is not None else FEATURE_COLUMNS
    x = df.copy()
    missing = []
    for c in cols:
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
    return x[cols]


def predict(model, features: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.Series:
    """Predict with a single model.
    
    Args:
        model: Trained model (XGBoost/LightGBM)
        features: Features DataFrame
        feature_cols: Feature columns to use. If None, uses FEATURE_COLUMNS.
    """
    x = _coerce_features(features, feature_cols)
    
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


def predict_ensemble(
    models: list, 
    weights: list[float] | None, 
    features: pd.DataFrame, 
    feature_cols: list[str] | None = None
) -> pd.Series:
    """Ensemble prediction with optional feature selection.
    
    Args:
        models: List of trained models
        weights: Model weights (None = equal weights)
        features: Features DataFrame
        feature_cols: Feature columns used during training. If None, uses FEATURE_COLUMNS.
    """
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
        out += w[i] * predict(m, features, feature_cols).astype(float).values
    return pd.Series(out, index=features.index, name="pred_return")


def predict_ensemble_with_uncertainty(
    models: list, 
    weights: list[float] | None, 
    features: pd.DataFrame,
    feature_cols: list[str] | None = None
) -> pd.DataFrame:
    """Return mean prediction, uncertainty (std across models), and confidence.
    
    Args:
        models: List of trained models
        weights: Model weights (None = equal weights)
        features: Features DataFrame
        feature_cols: Feature columns used during training. If None, uses FEATURE_COLUMNS.
    """
    if not models:
        raise ValueError("No models provided for ensemble prediction")
    
    # Get predictions from all models
    preds_list = [predict(m, features, feature_cols).values for m in models]
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


def predict_peak_days(
    peak_model, 
    features: pd.DataFrame,
    feature_cols: list[str] | None = None,
    min_days: int = 1,
    max_days: int = 10,
) -> pd.Series:
    """Predict optimal sell day (days from now until expected peak).
    
    Returns predicted days to peak, clipped to valid range.
    """
    if peak_model is None:
        # Default to mid-horizon if no model
        return pd.Series(5.0, index=features.index, name="pred_peak_days")
    
    preds = predict(peak_model, features, feature_cols).values.astype(float)
    
    # Clip to valid range and round to nearest day
    preds = np.clip(np.round(preds), min_days, max_days)
    
    return pd.Series(preds, index=features.index, name="pred_peak_days")


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


def save_ensemble(
    manifest_path: str | Path, 
    model_rel_paths: list[str], 
    model_types: list[str] | None = None, 
    weights: list[float] | None = None,
    peak_model_path: str | None = None,
) -> None:
    """Save ensemble manifest with model paths, types, and peak model."""
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "mixed_ensemble_v1",  # Support both XGBoost and LightGBM
        "models": model_rel_paths,
        "model_types": model_types or ["xgboost"] * len(model_rel_paths),
        "weights": weights,
        "peak_model": peak_model_path,  # Optional peak timing model
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


def load_ensemble(manifest_path: str | Path) -> tuple[list, list[float] | None, object | None]:
    """Load ensemble models and optional peak model.
    
    Returns (models, weights, peak_model) tuple.
    """
    mp = Path(manifest_path)
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    manifest_type = manifest.get("type")
    base = mp.parent
    
    # Load peak model if present
    peak_model = None
    peak_rel = manifest.get("peak_model")
    if peak_rel:
        try:
            peak_model = load_model(base / peak_rel, "lightgbm")
        except Exception:
            pass  # Peak model is optional
    
    # Support both old and new formats
    if manifest_type == "xgboost_ensemble_v1":
        _require_xgb()
        model_rel = manifest.get("models") or []
        weights = manifest.get("weights")
        models = [load_model(base / rel, "xgboost") for rel in model_rel]
        return models, weights, peak_model
    elif manifest_type == "mixed_ensemble_v1":
        model_rel = manifest.get("models") or []
        model_types = manifest.get("model_types") or ["xgboost"] * len(model_rel)
        weights = manifest.get("weights")
        models = [load_model(base / rel, mtype) for rel, mtype in zip(model_rel, model_types)]
        return models, weights, peak_model
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

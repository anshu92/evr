from __future__ import annotations

from typing import Iterable

import pandas as pd

from stock_screener.modeling.model import FEATURE_COLUMNS


_EXCLUDE_DEFAULT = {"sector_hash", "industry_hash", "is_tsx"}


def winsorize_mad(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    """Clip extreme values using MAD (Median Absolute Deviation)."""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0 or pd.isna(mad):
        return series
    lower = median - n_mad * mad
    upper = median + n_mad * mad
    return series.clip(lower=lower, upper=upper)


def zscore(series: pd.Series) -> pd.Series:
    """Z-score normalize a series."""
    mu = series.mean()
    sd = series.std()
    if sd == 0 or pd.isna(sd):
        return series * 0.0
    return (series - mu) / sd


def normalize_features_cross_section(
    df: pd.DataFrame,
    *,
    date_col: str | None,
    feature_cols: Iterable[str] | None = None,
    exclude_cols: Iterable[str] = _EXCLUDE_DEFAULT,
) -> pd.DataFrame:
    """Winsorize (MAD) and z-score features cross-sectionally.

    If date_col is provided, normalization is per-date. Otherwise, normalize once
    across the full DataFrame (single cross-section).
    """
    out = df.copy()
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLUMNS)
    exclude = set(exclude_cols)
    cols = [c for c in cols if c in out.columns and c not in exclude]

    if date_col and date_col in out.columns:
        for col in cols:
            if out[col].notna().sum() == 0:
                continue
            out[col] = out.groupby(date_col)[col].transform(winsorize_mad)
            out[col] = out.groupby(date_col)[col].transform(zscore)
    else:
        for col in cols:
            if out[col].notna().sum() == 0:
                continue
            out[col] = zscore(winsorize_mad(out[col]))
    return out


def build_calibration_map(
    actual_returns: pd.Series,
    n_quantiles: int = 20,
) -> dict:
    """Build quantile-based calibration map from training data.
    
    Returns a dict with quantile edges and corresponding return values
    that can be used to map predictions to realistic return magnitudes.
    """
    import numpy as np
    
    clean = actual_returns.dropna()
    if len(clean) < n_quantiles * 2:
        return {"quantiles": [], "values": [], "mean": float(clean.mean()), "std": float(clean.std())}
    
    # Compute quantile edges and values
    quantile_edges = np.linspace(0, 1, n_quantiles + 1)
    quantile_values = [float(clean.quantile(q)) for q in quantile_edges]
    
    return {
        "quantiles": quantile_edges.tolist(),
        "values": quantile_values,
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "n_samples": len(clean),
    }


def calibrate_predictions(
    predictions: pd.Series,
    calibration_map: dict,
    method: str = "rank_preserve",
) -> pd.Series:
    """Calibrate raw predictions to realistic return magnitudes.
    
    Args:
        predictions: Raw model predictions (any scale)
        calibration_map: Dict from build_calibration_map()
        method: 'rank_preserve' (default) or 'linear'
        
    Returns:
        Calibrated predictions matching the training return distribution
    """
    import numpy as np
    
    if not calibration_map or not calibration_map.get("values"):
        # Fallback: just center around training mean
        mean = calibration_map.get("mean", 0.0) if calibration_map else 0.0
        std = calibration_map.get("std", 1.0) if calibration_map else 1.0
        pred_mean = predictions.mean()
        pred_std = predictions.std()
        if pred_std > 0:
            return mean + (predictions - pred_mean) / pred_std * std
        return predictions
    
    quantiles = calibration_map["quantiles"]
    values = calibration_map["values"]
    
    if method == "rank_preserve":
        # Map prediction ranks to return quantiles (preserves ranking)
        ranks = predictions.rank(pct=True)
        
        # Interpolate to get calibrated values
        calibrated = np.interp(ranks, quantiles, values)
        return pd.Series(calibrated, index=predictions.index, name="pred_return_calibrated")
    
    elif method == "linear":
        # Linear scaling to match mean and std
        target_mean = calibration_map["mean"]
        target_std = calibration_map["std"]
        
        pred_mean = predictions.mean()
        pred_std = predictions.std()
        
        if pred_std > 0:
            scaled = (predictions - pred_mean) / pred_std * target_std + target_mean
        else:
            scaled = predictions - pred_mean + target_mean
        
        # Clip to reasonable bounds
        scaled = scaled.clip(calibration_map.get("min", -0.5), calibration_map.get("max", 0.5))
        return pd.Series(scaled, index=predictions.index, name="pred_return_calibrated")
    
    else:
        return predictions

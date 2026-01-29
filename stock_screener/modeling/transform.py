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

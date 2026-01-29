from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return pd.Series(0.0, index=series.index)
    mu = clean.mean()
    sd = clean.std()
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sd


def _zscore_xs(df: pd.DataFrame, series: pd.Series, *, date_col: str | None) -> pd.Series:
    if date_col and date_col in df.columns:
        return series.groupby(df[date_col]).transform(_zscore)
    return _zscore(series)


def add_fundamental_composites(df: pd.DataFrame, *, date_col: str | None) -> pd.DataFrame:
    """Add composite fundamental scores and interactions (optionally per-date)."""
    out = df.copy()

    value_components = []
    if "trailing_pe" in out.columns:
        inv_pe = 1.0 / out["trailing_pe"].replace([0, np.inf, -np.inf], np.nan)
        value_components.append(_zscore_xs(out, inv_pe, date_col=date_col))
    if "price_to_book" in out.columns:
        inv_pb = 1.0 / out["price_to_book"].replace([0, np.inf, -np.inf], np.nan)
        value_components.append(_zscore_xs(out, inv_pb, date_col=date_col))
    if "price_to_sales" in out.columns:
        inv_ps = 1.0 / out["price_to_sales"].replace([0, np.inf, -np.inf], np.nan)
        value_components.append(_zscore_xs(out, inv_ps, date_col=date_col))
    if value_components:
        out["value_score"] = pd.concat(value_components, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    else:
        out["value_score"] = 0.0

    quality_components = []
    if "return_on_equity" in out.columns:
        quality_components.append(_zscore_xs(out, out["return_on_equity"], date_col=date_col))
    if "operating_margins" in out.columns:
        quality_components.append(_zscore_xs(out, out["operating_margins"], date_col=date_col))
    if "profit_margins" in out.columns:
        quality_components.append(_zscore_xs(out, out["profit_margins"], date_col=date_col))
    if quality_components:
        out["quality_score"] = pd.concat(quality_components, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    else:
        out["quality_score"] = 0.0

    growth_components = []
    if "revenue_growth" in out.columns:
        growth_components.append(_zscore_xs(out, out["revenue_growth"], date_col=date_col))
    if "earnings_growth" in out.columns:
        growth_components.append(_zscore_xs(out, out["earnings_growth"], date_col=date_col))
    if growth_components:
        out["growth_score"] = pd.concat(growth_components, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    else:
        out["growth_score"] = 0.0

    if "target_mean_price" in out.columns and "last_close_cad" in out.columns:
        out["pe_discount"] = (out["target_mean_price"] - out["last_close_cad"]) / out["last_close_cad"].replace(
            0, np.nan
        )
        out["pe_discount"] = out["pe_discount"].replace([np.inf, -np.inf], np.nan)
    else:
        out["pe_discount"] = 0.0

    if "earnings_quarterly_growth" in out.columns:
        out["roc_growth"] = out["earnings_quarterly_growth"]
    else:
        out["roc_growth"] = 0.0

    if "value_score" in out.columns and "ret_120d" in out.columns:
        out["value_momentum"] = out["value_score"] * out["ret_120d"]
    else:
        out["value_momentum"] = 0.0

    if "vol_60d_ann" in out.columns and "log_market_cap" in out.columns:
        out["vol_size"] = out["vol_60d_ann"] * out["log_market_cap"]
    else:
        out["vol_size"] = 0.0

    if "quality_score" in out.columns and "growth_score" in out.columns:
        out["quality_growth"] = out["quality_score"] * out["growth_score"]
    else:
        out["quality_growth"] = 0.0

    return out

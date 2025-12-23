from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return s * 0.0
    return (s - mu) / sd


def screen_universe(
    features: pd.DataFrame,
    min_price_cad: float,
    min_avg_dollar_volume_cad: float,
    top_n: int,
    logger,
) -> pd.DataFrame:
    """Filter and rank tickers by a robust multi-factor score."""

    df = features.copy()

    # Basic data quality
    df = df[df["n_days"] >= 90]
    df = df.replace([np.inf, -np.inf], np.nan)

    # Liquidity + price filters (CAD base)
    df = df[df["last_close_cad"] >= float(min_price_cad)]
    df = df[df["avg_dollar_volume_cad"] >= float(min_avg_dollar_volume_cad)]

    if df.empty:
        raise RuntimeError("No tickers left after liquidity/price filters")

    # Score:
    # - If ML predictions are available, blend them in as the primary alpha term.
    # - Otherwise use a robust baseline factor score.
    has_ml = "pred_return" in df.columns and pd.to_numeric(df["pred_return"], errors="coerce").notna().any()

    baseline = (
        0.60 * _zscore(df["ret_60d"])
        + 0.35 * _zscore(df["ret_120d"])
        + 0.10 * _zscore(df["ma20_ratio"])
        + 0.10 * _zscore(np.log10(df["avg_dollar_volume_cad"].clip(lower=1.0)))
        - 0.35 * _zscore(df["vol_60d_ann"])
    )

    if has_ml:
        ml = _zscore(pd.to_numeric(df["pred_return"], errors="coerce"))
        score = 0.70 * ml + 0.30 * baseline
        logger.info("Scoring uses ML blend (70%% ML / 30%% baseline).")
    else:
        score = baseline
        logger.info("Scoring uses baseline factors (no ML predictions).")

    df["score"] = score
    df = df.sort_values("score", ascending=False)

    n = int(top_n)
    if n <= 0:
        n = 50
    out = df.head(n).copy()
    logger.info("Screened universe: %s tickers (from %s after filters)", len(out), len(df))
    return out



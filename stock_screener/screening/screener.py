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


def score_universe(
    features: pd.DataFrame,
    min_price_cad: float,
    min_avg_dollar_volume_cad: float,
    logger,
    *,
    max_volatility: float | None = None,
) -> pd.DataFrame:
    """Compute scores for all eligible tickers (no top-N truncation)."""

    df = features.copy()

    # Basic data quality
    df = df[df["n_days"] >= 90]
    df = df.replace([np.inf, -np.inf], np.nan)

    # Liquidity + price filters (CAD base)
    df = df[df["last_close_cad"] >= float(min_price_cad)]
    df = df[df["avg_dollar_volume_cad"] >= float(min_avg_dollar_volume_cad)]

    # Volatility cap — hard filter to exclude ultra-vol junk before scoring.
    # Without this, high-vol micro-caps dominate because peak_return ∝ vol.
    if max_volatility is not None and "vol_60d_ann" in df.columns:
        before = len(df)
        df = df[df["vol_60d_ann"] <= float(max_volatility)]
        dropped = before - len(df)
        if dropped > 0:
            logger.info("Vol cap (%.0f%%): removed %d tickers", max_volatility * 100, dropped)

    if df.empty:
        if logger:
            logger.warning("No tickers left after liquidity/price/vol filters")
        out = features.iloc[0:0].copy()
        out["score"] = pd.Series(dtype=float)
        return out

    # Score:
    # - If ML predictions are available, blend them in as the primary alpha term.
    # - Use risk-adjusted ret_per_day (return/day / vol) to prevent vol-chasing.
    # - Otherwise use a robust baseline factor score.
    has_score = "pred_score" in df.columns and pd.to_numeric(df["pred_score"], errors="coerce").notna().any()
    has_ret_per_day = "ret_per_day" in df.columns and pd.to_numeric(df["ret_per_day"], errors="coerce").notna().any()
    has_return = "pred_return" in df.columns and pd.to_numeric(df["pred_return"], errors="coerce").notna().any()
    has_ml = has_score or has_ret_per_day or has_return

    baseline = (
        0.60 * _zscore(df["ret_60d"])
        + 0.35 * _zscore(df["ret_120d"])
        + 0.10 * _zscore(df["ma20_ratio"])
        + 0.10 * _zscore(np.log10(df["avg_dollar_volume_cad"].clip(lower=1.0)))
        - 0.35 * _zscore(df["vol_60d_ann"])
    )

    if has_ml:
        # Priority: risk-adj ret_per_day > pred_score > pred_return
        # Risk-adjust by dividing by vol to prevent high-vol stocks from dominating.
        if has_ret_per_day:
            raw_signal = pd.to_numeric(df["ret_per_day"], errors="coerce")
            # Sharpe-like risk adjustment: return/day / vol
            if "vol_60d_ann" in df.columns:
                safe_vol = df["vol_60d_ann"].clip(lower=0.10)
                ml_signal = raw_signal / safe_vol
                ml_label = "risk_adj_ret_per_day (return/day/vol)"
            else:
                ml_signal = raw_signal
                ml_label = "ret_per_day (return/day)"
        elif has_score:
            ml_signal = df["pred_score"]
            ml_label = "pred_score"
        else:
            ml_signal = df["pred_return"]
            ml_label = "pred_return"
        ml = _zscore(pd.to_numeric(ml_signal, errors="coerce"))
        score = 0.60 * ml + 0.40 * baseline
        logger.info("Scoring uses ML blend: 60%% %s / 40%% baseline.", ml_label)
    else:
        score = baseline
        logger.info("Scoring uses baseline factors (no ML predictions).")

    df["score"] = score
    df = df.sort_values("score", ascending=False)
    return df


def select_sector_neutral(
    df: pd.DataFrame,
    top_n: int,
    sector_col: str = "sector",
    score_col: str = "score",
    min_per_sector: int = 1,
    max_per_sector: int | None = None,
) -> pd.DataFrame:
    """Select top stocks with sector diversification.
    
    Instead of simply taking top-N overall (which may concentrate in 1-2 sectors),
    this allocates picks across sectors proportionally, then fills remaining
    slots with the best overall picks.
    """
    n = max(1, int(top_n))
    
    # If no sector info, fall back to simple top-N
    if sector_col not in df.columns:
        return df.head(n)
    
    df = df.copy()
    df["_sector"] = df[sector_col].fillna("Unknown")
    
    # Count stocks per sector
    sector_counts = df.groupby("_sector").size()
    n_sectors = len(sector_counts)
    
    if n_sectors <= 1:
        return df.drop(columns=["_sector"]).head(n)
    
    # Calculate target allocation per sector (proportional to sector size)
    # Each sector gets at least min_per_sector, up to max_per_sector
    base_per_sector = max(min_per_sector, n // n_sectors)
    if max_per_sector is not None:
        base_per_sector = min(base_per_sector, max_per_sector)
    
    selected = []
    remaining_slots = n
    
    # First pass: pick top stocks from each sector
    for sector in sector_counts.index:
        sector_df = df[df["_sector"] == sector].sort_values(score_col, ascending=False)
        
        # Allocate proportionally to sector size
        sector_weight = sector_counts[sector] / len(df)
        sector_slots = max(min_per_sector, int(n * sector_weight))
        sector_slots = min(sector_slots, remaining_slots, len(sector_df))
        if max_per_sector is not None:
            sector_slots = min(sector_slots, max_per_sector)
        
        if sector_slots > 0:
            picks = sector_df.head(sector_slots)
            selected.append(picks)
            remaining_slots -= len(picks)
    
    if not selected:
        return df.drop(columns=["_sector"]).head(n)
    
    result = pd.concat(selected, ignore_index=False)
    
    # Second pass: fill remaining slots with best overall (not already selected)
    if remaining_slots > 0:
        already_picked = set(result.index)
        remaining = df[~df.index.isin(already_picked)].sort_values(score_col, ascending=False)
        fill = remaining.head(remaining_slots)
        result = pd.concat([result, fill], ignore_index=False)
    
    # Sort by score and return
    result = result.sort_values(score_col, ascending=False).drop(columns=["_sector"])
    return result.head(n)


def screen_universe(
    features: pd.DataFrame,
    min_price_cad: float,
    min_avg_dollar_volume_cad: float,
    top_n: int,
    logger,
    sector_neutral: bool = True,
    max_volatility: float | None = None,
) -> pd.DataFrame:
    """Filter and rank tickers by a robust multi-factor score."""
    scored = score_universe(
        features=features,
        min_price_cad=min_price_cad,
        min_avg_dollar_volume_cad=min_avg_dollar_volume_cad,
        logger=logger,
        max_volatility=max_volatility,
    )

    n = int(top_n)
    if n <= 0:
        n = 50
    
    if sector_neutral and "sector" in scored.columns:
        out = select_sector_neutral(scored, top_n=n, sector_col="sector", score_col="score")
        n_sectors = scored["sector"].nunique()
        logger.info(
            "Sector-neutral selection: %d tickers from %d sectors (from %d after filters)",
            len(out), out["sector"].nunique() if "sector" in out.columns else 0, len(scored)
        )
    else:
        out = scored.head(n).copy()
        logger.info("Screened universe: %s tickers (from %s after filters)", len(out), len(scored))
    
    return out


def apply_entry_filters(
    df: pd.DataFrame,
    *,
    min_confidence: float | None = None,
    min_pred_return: float | None = None,
    max_volatility: float | None = None,
    min_momentum_5d: float | None = None,
    momentum_alignment: bool = True,
    logger=None,
) -> tuple[pd.DataFrame, dict]:
    """Apply entry confirmation filters to reduce false positives.
    
    Returns:
        Tuple of (filtered DataFrame, stats dict)
    """
    original_count = len(df)
    filtered = df.copy()
    rejection_reasons = {}
    
    # Filter by model confidence
    if min_confidence is not None and "pred_confidence" in filtered.columns:
        mask = filtered["pred_confidence"] >= min_confidence
        rejected = (~mask).sum()
        if rejected > 0:
            rejection_reasons["low_confidence"] = int(rejected)
        filtered = filtered[mask]
    
    # Filter by predicted return (calibrated)
    if min_pred_return is not None and "pred_return" in filtered.columns:
        mask = filtered["pred_return"] >= min_pred_return
        rejected = (~mask).sum()
        if rejected > 0:
            rejection_reasons["low_pred_return"] = int(rejected)
        filtered = filtered[mask]
    
    # Filter by volatility (avoid highly volatile stocks)
    if max_volatility is not None and "vol_60d_ann" in filtered.columns:
        mask = filtered["vol_60d_ann"] <= max_volatility
        rejected = (~mask).sum()
        if rejected > 0:
            rejection_reasons["high_volatility"] = int(rejected)
        filtered = filtered[mask]
    
    # Filter by recent momentum (avoid stocks in freefall)
    if min_momentum_5d is not None and "ret_5d" in filtered.columns:
        mask = filtered["ret_5d"] >= min_momentum_5d
        rejected = (~mask).sum()
        if rejected > 0:
            rejection_reasons["negative_momentum"] = int(rejected)
        filtered = filtered[mask]
    
    # Momentum-price alignment: reject if model is bullish but price trend is strongly bearish
    # This helps avoid "catching falling knives"
    if momentum_alignment and "pred_return" in filtered.columns and "ret_5d" in filtered.columns:
        # For stocks with positive prediction, require 5d momentum not severely negative
        # Threshold: if predicting +X%, allow momentum down to -2X%
        pred_ret = filtered["pred_return"].fillna(0)
        ret_5d = filtered["ret_5d"].fillna(0)
        
        # Calculate alignment threshold: more bullish prediction allows slightly worse momentum
        alignment_threshold = -2.0 * pred_ret.clip(lower=0)
        alignment_threshold = alignment_threshold.clip(lower=-0.10)  # Max -10% floor
        
        # Misalignment: bullish prediction (>0) but momentum far below threshold
        misaligned = (pred_ret > 0.005) & (ret_5d < alignment_threshold)
        rejected = misaligned.sum()
        if rejected > 0:
            rejection_reasons["momentum_misalignment"] = int(rejected)
        filtered = filtered[~misaligned]
    
    stats = {
        "original_count": original_count,
        "filtered_count": len(filtered),
        "rejected_count": original_count - len(filtered),
        "rejection_reasons": rejection_reasons,
    }
    
    if logger and stats["rejected_count"] > 0:
        logger.info(
            "Entry filters: %d -> %d stocks (rejected: %s)",
            original_count, len(filtered), rejection_reasons
        )
    
    return filtered, stats

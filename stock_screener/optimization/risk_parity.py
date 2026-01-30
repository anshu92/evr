from __future__ import annotations

import numpy as np
import pandas as pd


def _cap_weights(w: pd.Series, cap: float, *, allow_cash: bool) -> pd.Series:
    cap = float(cap)
    if cap <= 0 or cap >= 1:
        return w / w.sum()
    w = w.copy()
    for _ in range(10):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w.loc[over] = cap
        under = ~over
        if not under.any():
            break
        under_sum = float(w.loc[under].sum())
        if under_sum > 0:
            w.loc[under] = w.loc[under] + (w.loc[under] / under_sum) * excess
        else:
            # Under-cap names all had weight 0 (alpha col zero/NaN). Distribute excess equally to avoid 0/0 → NaN.
            n_under = int(under.sum())
            if n_under > 0:
                w.loc[under] = excess / n_under
    if allow_cash:
        return w
    total = float(w.sum())
    return w / total if total > 0 else w


def compute_inverse_vol_weights(
    features: pd.DataFrame,
    portfolio_size: int,
    weight_cap: float,
    logger,
    *,
    alpha_col: str | None = None,
    alpha_floor: float = 0.0,
) -> pd.DataFrame:
    """Compute inverse-vol (risk parity style) weights from screened features."""

    df = features.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    n = int(portfolio_size)
    if n <= 0:
        n = 20
    df = df.head(n).copy()

    vol = pd.to_numeric(df["vol_60d_ann"], errors="coerce")
    vol = vol.replace([0.0, np.inf, -np.inf], np.nan).dropna()
    df = df.loc[vol.index].copy()
    if df.empty:
        raise RuntimeError("No tickers with valid volatility for weighting")

    inv = 1.0 / vol
    if alpha_col and alpha_col in df.columns:
        alpha = pd.to_numeric(df[alpha_col], errors="coerce").reindex(vol.index)
        alpha = alpha.clip(lower=float(alpha_floor)).fillna(0.0)
        if float(alpha.sum()) > 0:
            inv = inv * alpha
            logger.info("Applied alpha weighting using %s (floor=%s).", alpha_col, alpha_floor)
        else:
            logger.warning("Alpha column %s has no positive values; falling back to pure inverse-vol.", alpha_col)
    inv_sum = float(inv.sum())
    if not np.isfinite(inv_sum) or inv_sum <= 0:
        logger.warning("Inverse-vol sum invalid (sum=%s); falling back to equal weights.", inv_sum)
        inv = pd.Series(1.0, index=inv.index)
        inv_sum = float(inv.sum())
    w = inv / inv_sum
    cap = float(weight_cap)
    allow_cash = False
    if 0 < cap < 1 and cap * len(w) < 1:
        allow_cash = True
        logger.warning(
            "Weight cap %.3f with %s names implies max invested %.1f%%; leaving cash unallocated.",
            cap,
            len(w),
            cap * len(w) * 100.0,
        )
    w = _cap_weights(w, cap, allow_cash=allow_cash)
    w = w.fillna(0.0)
    if w.sum() > 0:
        w = w / w.sum()

    out = df.copy()
    out["weight"] = w
    out = out.sort_values("weight", ascending=False)
    logger.info("Computed weights: %s tickers (cap=%s)", len(out), weight_cap)
    return out

from __future__ import annotations

import numpy as np
import pandas as pd


def _cap_weights(w: pd.Series, cap: float) -> pd.Series:
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
        w.loc[under] = w.loc[under] + (w.loc[under] / float(w.loc[under].sum())) * excess
    return w / float(w.sum())


def compute_inverse_vol_weights(
    features: pd.DataFrame,
    portfolio_size: int,
    weight_cap: float,
    logger,
) -> pd.DataFrame:
    """Compute inverse-vol (risk parity style) weights from screened features."""

    df = features.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    n = int(portfolio_size)
    if n <= 0:
        n = 20
    df = df.head(n).copy()

    vol = pd.to_numeric(df["vol_60d_ann"], errors="coerce")
    vol = vol.replace(0.0, np.nan).dropna()
    df = df.loc[vol.index].copy()
    if df.empty:
        raise RuntimeError("No tickers with valid volatility for weighting")

    inv = 1.0 / vol
    w = inv / float(inv.sum())
    w = _cap_weights(w, float(weight_cap))

    out = df.copy()
    out["weight"] = w
    out = out.sort_values("weight", ascending=False)
    logger.info("Computed weights: %s tickers (cap=%s)", len(out), weight_cap)
    return out



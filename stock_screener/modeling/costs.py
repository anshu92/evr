from __future__ import annotations

import numpy as np
import pandas as pd


def apply_cost_to_label(label: pd.Series, cost_bps: pd.Series) -> pd.Series:
    """Convert a gross return label to a net return label using per-row cost in bps."""
    lbl = pd.to_numeric(label, errors="coerce")
    bps = pd.to_numeric(cost_bps, errors="coerce").fillna(0.0)
    return lbl - (bps * 1e-4)


def estimate_trade_cost_bps(
    features: pd.DataFrame,
    *,
    date_col: str = "date",
    base_bps: float = 3.0,
    spread_coef: float = 0.5,
    vol_coef: float = 0.5,
    min_bps: float = 0.5,
    max_bps: float = 100.0,
) -> pd.Series:
    """Estimate per-sample round-trip trade cost (bps) from liquidity/volatility proxies.

    The model is intentionally lightweight and robust:
    - Baseline cost: `base_bps`
    - Illiquidity add-on: cross-sectional rank of inverse ADV
    - Volatility add-on: cross-sectional rank of annualized volatility
    """
    if features.empty:
        return pd.Series(dtype=float)

    out = pd.Series(float(base_bps), index=features.index, dtype=float)
    same_day_group = features[date_col] if date_col in features.columns else None

    if "avg_dollar_volume_cad" in features.columns:
        adv = pd.to_numeric(features["avg_dollar_volume_cad"], errors="coerce")
        if same_day_group is not None:
            liq_rank = adv.groupby(same_day_group).rank(pct=True, ascending=True)
        else:
            liq_rank = adv.rank(pct=True, ascending=True)
        illiq = (1.0 - liq_rank).fillna(0.5).clip(0.0, 1.0)
        out = out + float(base_bps) * float(spread_coef) * illiq

    vol_col = "vol_20d_ann" if "vol_20d_ann" in features.columns else (
        "vol_60d_ann" if "vol_60d_ann" in features.columns else None
    )
    if vol_col is not None:
        vol = pd.to_numeric(features[vol_col], errors="coerce")
        if same_day_group is not None:
            vol_rank = vol.groupby(same_day_group).rank(pct=True, ascending=True)
        else:
            vol_rank = vol.rank(pct=True, ascending=True)
        vol_rank = vol_rank.fillna(0.5).clip(0.0, 1.0)
        out = out + float(base_bps) * float(vol_coef) * vol_rank

    out = out.replace([np.inf, -np.inf], np.nan).fillna(float(base_bps))
    return out.clip(lower=float(min_bps), upper=float(max_bps))

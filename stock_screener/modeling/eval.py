from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_dates: list[pd.Timestamp]
    val_dates: list[pd.Timestamp]


@dataclass(frozen=True)
class HoldoutSplit:
    train_dates: list[pd.Timestamp]
    holdout_dates: list[pd.Timestamp]


def _as_dates(dates: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
    out = [pd.to_datetime(d).normalize() for d in dates]
    return sorted(list(dict.fromkeys(out)))


def build_time_splits(
    dates: Iterable[pd.Timestamp],
    *,
    n_splits: int = 3,
    val_window: int = 60,
    embargo_days: int = 5,
    min_train: int = 252,
) -> tuple[list[TimeSplit], HoldoutSplit]:
    """Create expanding-window time splits with an embargo and final holdout."""

    uniq = _as_dates(dates)
    if len(uniq) < min_train + val_window + embargo_days + 5:
        raise ValueError("Not enough dates to build time splits")

    splits: list[TimeSplit] = []
    for i in range(n_splits):
        train_end = min_train + i * val_window
        val_start = train_end + embargo_days
        val_end = val_start + val_window
        if val_end > len(uniq):
            break
        splits.append(TimeSplit(train_dates=uniq[:train_end], val_dates=uniq[val_start:val_end]))

    holdout_end = len(uniq)
    holdout_start = max(0, holdout_end - val_window)
    train_end = max(0, holdout_start - embargo_days)
    holdout = HoldoutSplit(train_dates=uniq[:train_end], holdout_dates=uniq[holdout_start:holdout_end])
    return splits, holdout


def compute_daily_rank_ic(
    df: pd.DataFrame,
    *,
    date_col: str,
    label_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """Compute per-date Spearman rank IC for predictions."""

    if df.empty:
        return pd.DataFrame(columns=["date", "n", "rank_ic"])

    frame = df[[date_col, label_col, pred_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    frame = frame.dropna(subset=[label_col, pred_col])

    rows: list[dict[str, float]] = []
    for dt, grp in frame.groupby(date_col):
        if len(grp) < 2:
            continue
        ic = grp[label_col].corr(grp[pred_col], method="spearman")
        if pd.isna(ic):
            continue
        rows.append({"date": dt, "n": float(len(grp)), "rank_ic": float(ic)})

    return pd.DataFrame(rows)


def compute_daily_topn_returns(
    df: pd.DataFrame,
    *,
    date_col: str,
    label_col: str,
    pred_col: str,
    top_n: int,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Compute per-date top-N mean returns with optional per-day cost in bps."""

    if df.empty:
        return pd.DataFrame(columns=["date", "n", "mean_ret", "net_ret"])

    frame = df[[date_col, label_col, pred_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    frame = frame.dropna(subset=[label_col, pred_col])

    n = max(1, int(top_n))
    cost = float(cost_bps) * 1e-4

    rows: list[dict[str, float]] = []
    for dt, grp in frame.groupby(date_col):
        if len(grp) < 1:
            continue
        top = grp.sort_values(pred_col, ascending=False).head(n)
        mean_ret = float(top[label_col].mean())
        net_ret = float(mean_ret - cost)
        rows.append({"date": dt, "n": float(len(top)), "mean_ret": mean_ret, "net_ret": net_ret})

    return pd.DataFrame(rows)


def summarize_rank_ic(daily_ic: pd.DataFrame) -> dict[str, float]:
    if daily_ic.empty:
        return {"mean_ic": float("nan"), "std_ic": float("nan"), "ic_ir": float("nan"), "n_days": 0}
    mean_ic = float(daily_ic["rank_ic"].mean())
    std_ic = float(daily_ic["rank_ic"].std(ddof=0))
    ic_ir = float(mean_ic / std_ic) if std_ic > 0 else float("nan")
    return {"mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": ic_ir, "n_days": int(len(daily_ic))}


def summarize_topn_returns(daily_ret: pd.DataFrame) -> dict[str, float]:
    if daily_ret.empty:
        return {
            "mean_ret": float("nan"),
            "std_ret": float("nan"),
            "ret_ir": float("nan"),
            "mean_net_ret": float("nan"),
            "std_net_ret": float("nan"),
            "net_ret_ir": float("nan"),
            "n_days": 0,
        }
    mean_ret = float(daily_ret["mean_ret"].mean())
    std_ret = float(daily_ret["mean_ret"].std(ddof=0))
    ret_ir = float(mean_ret / std_ret) if std_ret > 0 else float("nan")
    mean_net = float(daily_ret["net_ret"].mean())
    std_net = float(daily_ret["net_ret"].std(ddof=0))
    net_ir = float(mean_net / std_net) if std_net > 0 else float("nan")
    return {
        "mean_ret": mean_ret,
        "std_ret": std_ret,
        "ret_ir": ret_ir,
        "mean_net_ret": mean_net,
        "std_net_ret": std_net,
        "net_ret_ir": net_ir,
        "n_days": int(len(daily_ret)),
    }


def evaluate_topn_returns(
    df: pd.DataFrame,
    *,
    date_col: str,
    label_col: str,
    pred_col: str,
    top_n: int,
    cost_bps: float = 0.0,
) -> dict[str, object]:
    """Evaluate top-N realized returns across dates."""

    daily = compute_daily_topn_returns(
        df, date_col=date_col, label_col=label_col, pred_col=pred_col, top_n=top_n, cost_bps=cost_bps
    )
    summary = summarize_topn_returns(daily)
    return {"summary": summary, "daily": daily}


def evaluate_predictions(
    df: pd.DataFrame,
    *,
    date_col: str,
    label_col: str,
    pred_col: str,
    group_col: str | None = None,
) -> dict[str, object]:
    """Evaluate rank IC across dates; optionally include group breakdowns."""

    daily = compute_daily_rank_ic(df, date_col=date_col, label_col=label_col, pred_col=pred_col)
    summary = summarize_rank_ic(daily)
    payload: dict[str, object] = {"summary": summary, "daily": daily}

    if group_col and group_col in df.columns:
        group_stats: dict[str, dict[str, float]] = {}
        for g, grp in df.groupby(group_col):
            g_daily = compute_daily_rank_ic(grp, date_col=date_col, label_col=label_col, pred_col=pred_col)
            group_stats[str(g)] = summarize_rank_ic(g_daily)
        payload["by_group"] = group_stats

    return payload

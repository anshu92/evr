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


def summarize_rank_ic(daily_ic: pd.DataFrame) -> dict[str, float]:
    if daily_ic.empty:
        return {"mean_ic": float("nan"), "std_ic": float("nan"), "ic_ir": float("nan"), "n_days": 0}
    mean_ic = float(daily_ic["rank_ic"].mean())
    std_ic = float(daily_ic["rank_ic"].std(ddof=0))
    ic_ir = float(mean_ic / std_ic) if std_ic > 0 else float("nan")
    return {"mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": ic_ir, "n_days": int(len(daily_ic))}


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

"""Realized-return feedback engine -- online IC, ensemble reweighting, verified labels."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from stock_screener.reward.tracker import RewardLog


def compute_online_ic(
    reward_log: RewardLog,
    *,
    window: int = 20,
) -> dict[str, Any]:
    """Compute Spearman rank IC of predicted vs realized returns.

    Returns dict with 'ensemble_ic', 'per_model_ics' (list of per-model IC),
    and 'n_observations'.
    """
    entries = reward_log.entries_for_window(last_n_days=window)
    # Need entries that have both a prediction and a realized return
    valid = [
        e for e in entries
        if e.realized_1d_return is not None and e.predicted_return is not None
    ]
    if len(valid) < 5:
        return {"ensemble_ic": None, "per_model_ics": None, "n_observations": len(valid)}

    pred = np.array([e.predicted_return for e in valid])
    real = np.array([e.realized_1d_return for e in valid])

    # Ensemble-level IC
    ensemble_ic = float(spearmanr(pred, real).statistic)

    # Per-model ICs (if per_model_preds are stored)
    per_model_ics: list[float] | None = None
    sample_with_models = [e for e in valid if e.per_model_preds is not None and len(e.per_model_preds) > 0]
    if len(sample_with_models) >= 5:
        n_models = len(sample_with_models[0].per_model_preds)  # type: ignore[arg-type]
        # Only use entries that have the expected number of per-model preds
        usable = [e for e in sample_with_models if len(e.per_model_preds) == n_models]  # type: ignore[arg-type]
        if len(usable) >= 5:
            real_arr = np.array([e.realized_1d_return for e in usable])
            per_model_ics = []
            for i in range(n_models):
                model_preds = np.array([e.per_model_preds[i] for e in usable])  # type: ignore[index]
                ic = float(spearmanr(model_preds, real_arr).statistic)
                per_model_ics.append(ic)

    return {
        "ensemble_ic": ensemble_ic,
        "per_model_ics": per_model_ics,
        "n_observations": len(valid),
    }


def compute_ensemble_reward_weights(
    per_model_ics: list[float],
    holdout_weights: list[float] | None,
    *,
    blend_alpha: float = 0.5,
    ic_decay: float = 0.9,
) -> list[float]:
    """Blend realized IC-based weights with original holdout IC weights.

    blend_alpha controls the mix: 1.0 = fully realized IC, 0.0 = fully holdout.
    """
    n = len(per_model_ics)
    # Realized IC weights: proportional to IC, floored at 0
    realized_raw = np.array([max(ic, 0.0) for ic in per_model_ics], dtype=float)
    total = realized_raw.sum()
    if total > 0:
        realized_w = realized_raw / total
    else:
        realized_w = np.ones(n, dtype=float) / n

    # Holdout weights (fallback: equal)
    if holdout_weights is not None and len(holdout_weights) == n:
        holdout_w = np.array(holdout_weights, dtype=float)
        total_h = holdout_w.sum()
        if total_h > 0:
            holdout_w = holdout_w / total_h
        else:
            holdout_w = np.ones(n, dtype=float) / n
    else:
        holdout_w = np.ones(n, dtype=float) / n

    blended = blend_alpha * realized_w + (1.0 - blend_alpha) * holdout_w
    blended = blended / blended.sum()
    return blended.tolist()


def build_verified_labels(
    reward_log: RewardLog,
) -> pd.DataFrame:
    """Extract closed trades as verified (ticker, date, realized_return) rows.

    These can be used as high-confidence training samples during retraining.
    """
    closed = reward_log.closed_entries()
    if not closed:
        return pd.DataFrame(columns=["date", "ticker", "realized_return", "days_held", "exit_reason"])

    rows = []
    for e in closed:
        ret = e.realized_cumulative_return if e.realized_cumulative_return is not None else e.realized_1d_return
        if ret is None:
            continue
        rows.append({
            "date": e.date,
            "ticker": e.ticker,
            "realized_return": ret,
            "days_held": e.days_held,
            "exit_reason": e.exit_reason,
        })
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "realized_return", "days_held", "exit_reason"])
    return pd.DataFrame(rows)


def compute_prediction_bias(
    reward_log: RewardLog,
    *,
    window: int = 30,
) -> dict[str, float]:
    """Detect systematic over/under-prediction.

    Returns dict with 'mean_bias' (pred - realized) and 'bias_std'.
    A positive bias means the model consistently over-predicts.
    """
    entries = reward_log.entries_for_window(last_n_days=window)
    valid = [
        e for e in entries
        if e.realized_1d_return is not None and e.predicted_return is not None
    ]
    if len(valid) < 5:
        return {"mean_bias": 0.0, "bias_std": 0.0, "n_observations": len(valid)}

    biases = np.array([e.predicted_return - e.realized_1d_return for e in valid])
    return {
        "mean_bias": float(np.mean(biases)),
        "bias_std": float(np.std(biases)),
        "n_observations": len(valid),
    }

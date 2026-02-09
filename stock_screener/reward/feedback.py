"""Realized-return feedback engine -- online IC, ensemble reweighting, verified labels, action scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from stock_screener.reward.tracker import RewardLog, ActionRewardLog, ActionRewardEntry


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


# ---------------------------------------------------------------------------
# Per-action reward scoring — weighted composite approach
# ---------------------------------------------------------------------------

# Component weights for the composite score (sum to 1.0)
_DEFAULT_WEIGHTS = {
    "base_outcome": 0.30,        # Raw trade P&L or post-action return
    "opportunity_cost": 0.20,    # Premature sell / missed upside penalty
    "rotation_alpha": 0.15,      # Did the replacement stock outperform?
    "selection_quality": 0.10,   # Did we pick the best available stock?
    "confidence_calibration": 0.10,  # Penalize high-conf mistakes
    "vol_adjusted": 0.10,        # Risk-adjusted quality of the action
    "timing_bonus": 0.05,        # Bonus for exit near top / entry near bottom
}


@dataclass
class ActionScoreParams:
    """Tunable parameters for the composite scoring engine."""

    premature_sell_threshold: float = 0.03
    bad_buy_threshold: float = -0.03
    wrong_hold_threshold: float = -0.05
    # Opportunity cost: how much to penalize missed upside after sell
    opp_cost_multiplier: float = 2.0
    # Correct-sell bonus: fraction of post-sell decline to credit
    correct_sell_bonus_frac: float = 0.5
    # Component weights (override defaults if provided)
    component_weights: dict[str, float] | None = None

    @property
    def weights(self) -> dict[str, float]:
        return self.component_weights or _DEFAULT_WEIGHTS


def score_action(
    entry: ActionRewardEntry,
    *,
    params: ActionScoreParams | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute a composite reward and per-component breakdown.

    Returns (total_reward, components_dict).  Each component is a raw
    score (not yet weighted); the composite is the weighted sum.

    Components:
      base_outcome         – realised trade return or post-action return
      opportunity_cost     – penalty for premature sell / reward for correct exit
      rotation_alpha       – replacement stock vs. sold stock performance
      selection_quality    – our pick vs. screened average
      confidence_calibration – penalise high-conf mistakes, reward high-conf wins
      vol_adjusted         – return normalised by expected volatility
      timing_bonus         – bonus for selling near subsequent peak / buying near dip
    """
    p = params or ActionScoreParams()
    post_ret = _best_post_return(entry)

    components: dict[str, float] = {k: 0.0 for k in p.weights}

    if entry.action in ("SELL", "SELL_PARTIAL"):
        _score_sell(entry, post_ret, p, components)
    elif entry.action == "BUY":
        _score_buy(entry, post_ret, p, components)
    elif entry.action == "HOLD":
        _score_hold(entry, post_ret, p, components)

    # ---- Shared cross-action components ----

    # Rotation alpha (from backfill)
    if entry.rotation_alpha is not None:
        # Positive = replacement did better (good rotation)
        components["rotation_alpha"] = float(entry.rotation_alpha)

    # Selection quality
    if entry.selection_alpha_5d is not None:
        components["selection_quality"] = float(entry.selection_alpha_5d)

    # Confidence calibration
    if entry.confidence_calibration is not None:
        components["confidence_calibration"] = float(entry.confidence_calibration)
    elif post_ret is not None and entry.confidence and entry.confidence > 0:
        # Fallback: compute inline if backfill didn't run
        pred_correct = (post_ret >= 0) == (entry.predicted_return >= 0)
        components["confidence_calibration"] = entry.confidence if pred_correct else -entry.confidence

    # Vol-adjusted return
    if entry.vol_adjusted_return_5d is not None:
        # Normalize to a similar magnitude as other components (~0.01-0.10 scale)
        components["vol_adjusted"] = float(entry.vol_adjusted_return_5d) * 0.01
    elif post_ret is not None and entry.stock_volatility and entry.stock_volatility > 0:
        expected_5d_vol = entry.stock_volatility * np.sqrt(5.0 / 252.0)
        if expected_5d_vol > 1e-6:
            components["vol_adjusted"] = (post_ret / expected_5d_vol) * 0.01

    # ---- Weighted composite ----
    total = sum(p.weights.get(k, 0.0) * v for k, v in components.items())
    return total, components


def _score_sell(
    entry: ActionRewardEntry,
    post_ret: float | None,
    p: ActionScoreParams,
    components: dict[str, float],
) -> None:
    """Populate score components for SELL / SELL_PARTIAL."""
    # Base outcome: realised trade return
    base_gain = 0.0
    if entry.entry_price and entry.entry_price > 0 and entry.price_at_action > 0:
        base_gain = (entry.price_at_action - entry.entry_price) / entry.entry_price
    components["base_outcome"] = base_gain

    if post_ret is None:
        return

    # Opportunity cost: penalty for premature sell, bonus for correct sell
    opp_cost = max(0.0, post_ret - p.premature_sell_threshold)
    correct_sell_bonus = max(0.0, -post_ret) * p.correct_sell_bonus_frac
    components["opportunity_cost"] = correct_sell_bonus - p.opp_cost_multiplier * opp_cost

    # Timing: sold and stock went down → good timing (scaled 0..1)
    if post_ret < 0:
        # Stock dropped after sell: good timing proportional to avoidance
        components["timing_bonus"] = min(1.0, abs(post_ret) / 0.05)
    else:
        # Stock rose after sell: bad timing
        components["timing_bonus"] = -min(1.0, post_ret / 0.05)


def _score_buy(
    entry: ActionRewardEntry,
    post_ret: float | None,
    p: ActionScoreParams,
    components: dict[str, float],
) -> None:
    """Populate score components for BUY."""
    if post_ret is None:
        return

    # Base outcome: post-buy return
    components["base_outcome"] = post_ret
    # Extra penalty for especially bad buys
    if post_ret < p.bad_buy_threshold:
        components["opportunity_cost"] = post_ret - p.bad_buy_threshold

    # Timing: bought near subsequent low → good timing
    if post_ret > 0:
        # Stock went up after buy: good entry
        components["timing_bonus"] = min(1.0, post_ret / 0.05)
    else:
        # Stock went down after buy: bad entry
        components["timing_bonus"] = max(-1.0, post_ret / 0.05)


def _score_hold(
    entry: ActionRewardEntry,
    post_ret: float | None,
    p: ActionScoreParams,
    components: dict[str, float],
) -> None:
    """Populate score components for HOLD."""
    if post_ret is None:
        return

    components["base_outcome"] = post_ret

    # Penalty for holding through steep decline
    if post_ret < p.wrong_hold_threshold:
        components["opportunity_cost"] = post_ret - p.wrong_hold_threshold

    # Bonus for patient hold through drawdown that recovered
    # If 1d was negative but 5d is positive → patience rewarded
    if entry.return_1d is not None and entry.return_5d is not None:
        if entry.return_1d < 0 and entry.return_5d > 0:
            components["timing_bonus"] = min(1.0, entry.return_5d / 0.03) * 0.5

    # Penalty for holding when alternatives were better
    if entry.screened_top_pred_return is not None and entry.predicted_return is not None:
        gap = entry.screened_top_pred_return - entry.predicted_return
        if gap > 0.02:
            # We held a weaker stock when a stronger one was available
            components["selection_quality"] = -gap


def score_actions(
    action_log: ActionRewardLog,
    *,
    window: int = 30,
    params: ActionScoreParams | None = None,
) -> int:
    """Score all un-scored action entries that have post-action data.

    Writes action_reward and reward_components back into each entry.
    Returns number scored.
    """
    p = params or ActionScoreParams()
    entries = action_log.entries_for_window(last_n_days=window)
    n_scored = 0
    for e in entries:
        if e.action_reward is not None:
            continue
        if _best_post_return(e) is None:
            continue
        total, comps = score_action(e, params=p)
        e.action_reward = total
        e.reward_components = comps
        n_scored += 1
    return n_scored


def compute_action_quality_summary(
    action_log: ActionRewardLog,
    *,
    window: int = 30,
) -> dict[str, Any]:
    """Compute aggregate quality metrics per action type.

    Includes per-component averages, classification counts, and rotation stats.
    """
    entries = action_log.entries_for_window(last_n_days=window)
    scored = [e for e in entries if e.action_reward is not None]
    if not scored:
        return {}

    summary: dict[str, Any] = {}
    for action_type in ("BUY", "SELL", "SELL_PARTIAL", "HOLD"):
        group = [e for e in scored if e.action == action_type]
        if not group:
            continue
        rewards = [e.action_reward for e in group]  # type: ignore[arg-type]
        info: dict[str, Any] = {
            "count": len(group),
            "avg_reward": float(np.mean(rewards)),
            "median_reward": float(np.median(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "positive_pct": float(np.mean([r > 0 for r in rewards])),
        }

        # Per-component averages
        comp_sums: dict[str, list[float]] = {}
        for e in group:
            if e.reward_components:
                for k, v in e.reward_components.items():
                    comp_sums.setdefault(k, []).append(v)
        if comp_sums:
            info["avg_components"] = {
                k: float(np.mean(vs)) for k, vs in comp_sums.items()
            }

        if action_type in ("SELL", "SELL_PARTIAL"):
            premature = [e for e in group if e.was_premature_sell is True]
            info["premature_sell_count"] = len(premature)
            info["premature_sell_pct"] = len(premature) / len(group) if group else 0.0
            if premature:
                info["avg_missed_upside"] = float(np.mean([
                    e.post_sell_return_5d for e in premature if e.post_sell_return_5d is not None
                ])) if any(e.post_sell_return_5d is not None for e in premature) else 0.0
            # Rotation stats
            with_rotation = [e for e in group if e.rotation_alpha is not None]
            if with_rotation:
                alphas = [e.rotation_alpha for e in with_rotation]  # type: ignore[arg-type]
                info["rotation_count"] = len(with_rotation)
                info["avg_rotation_alpha"] = float(np.mean(alphas))
                info["good_rotation_pct"] = float(np.mean([a > 0 for a in alphas]))

        elif action_type == "BUY":
            bad = [e for e in group if e.was_bad_buy is True]
            info["bad_buy_count"] = len(bad)
            info["bad_buy_pct"] = len(bad) / len(group) if group else 0.0
            if bad:
                info["avg_bad_buy_loss"] = float(np.mean([
                    e.buy_outcome_return for e in bad if e.buy_outcome_return is not None
                ])) if any(e.buy_outcome_return is not None for e in bad) else 0.0
            # Rotation stats (bought to replace a sold stock)
            with_rotation = [e for e in group if e.rotation_alpha is not None]
            if with_rotation:
                alphas = [e.rotation_alpha for e in with_rotation]  # type: ignore[arg-type]
                info["rotation_count"] = len(with_rotation)
                info["avg_rotation_alpha"] = float(np.mean(alphas))
                info["good_rotation_pct"] = float(np.mean([a > 0 for a in alphas]))

        elif action_type == "HOLD":
            wrong = [e for e in group if e.was_wrong_hold is True]
            info["wrong_hold_count"] = len(wrong)
            info["wrong_hold_pct"] = len(wrong) / len(group) if group else 0.0
            if wrong:
                info["avg_hold_loss"] = float(np.mean([
                    e.hold_period_return for e in wrong if e.hold_period_return is not None
                ])) if any(e.hold_period_return is not None for e in wrong) else 0.0
            # Patient hold stats
            patient = [e for e in group
                       if e.return_1d is not None and e.return_5d is not None
                       and e.return_1d < 0 and e.return_5d > 0]
            if patient:
                info["patient_hold_count"] = len(patient)
                info["patient_hold_pct"] = len(patient) / len(group)
                info["avg_patient_recovery"] = float(np.mean([e.return_5d for e in patient]))  # type: ignore[arg-type]

        summary[action_type] = info

    # Overall action accuracy
    all_rewards = [e.action_reward for e in scored]  # type: ignore[arg-type]
    summary["overall"] = {
        "total_actions_scored": len(scored),
        "avg_reward": float(np.mean(all_rewards)),
        "positive_action_pct": float(np.mean([r > 0 for r in all_rewards])),
    }

    # Overall component contribution
    all_comps: dict[str, list[float]] = {}
    for e in scored:
        if e.reward_components:
            for k, v in e.reward_components.items():
                all_comps.setdefault(k, []).append(v)
    if all_comps:
        summary["overall"]["avg_components"] = {
            k: float(np.mean(vs)) for k, vs in all_comps.items()
        }

    # Confidence calibration summary
    cal_entries = [e for e in scored if e.confidence_calibration is not None]
    if cal_entries:
        cals = [e.confidence_calibration for e in cal_entries]  # type: ignore[arg-type]
        summary["overall"]["confidence_calibration"] = {
            "avg": float(np.mean(cals)),
            "well_calibrated_pct": float(np.mean([c > 0 for c in cals])),
        }

    return summary


def _best_post_return(entry: ActionRewardEntry) -> float | None:
    """Return the best available post-action return (5d > 3d > 1d)."""
    if entry.return_5d is not None:
        return entry.return_5d
    if entry.return_3d is not None:
        return entry.return_3d
    if entry.return_1d is not None:
        return entry.return_1d
    return None

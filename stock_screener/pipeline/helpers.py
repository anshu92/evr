from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from stock_screener.config import Config


def compute_dynamic_portfolio_size(
    screened: pd.DataFrame,
    min_confidence: float,
    min_pred_return: float,
    max_positions: int,
    model_ic: float | None,
    logger,
) -> int:
    """Compute fully dynamic portfolio size based on model metrics and predicted returns.

    No base size - portfolio can be 1 to max_positions based entirely on:
    - Number of stocks meeting confidence and return thresholds
    - Model IC (if available) to scale aggressiveness
    - Quality score combining confidence and predicted return
    """
    if screened.empty:
        logger.warning("Dynamic sizing: No stocks to evaluate, defaulting to 1")
        return 1

    has_confidence = "pred_confidence" in screened.columns
    has_return = "pred_return" in screened.columns

    if not has_confidence and not has_return:
        # Fallback: use score percentile to pick top performers
        if "score" in screened.columns:
            # Take stocks with score > 75th percentile, at least 1
            threshold = screened["score"].quantile(0.75)
            count = max(1, min(max_positions, (screened["score"] >= threshold).sum()))
            logger.info("Dynamic sizing: No ML metrics, using score threshold, selecting %d positions", count)
            return count
        fallback = max(1, min(max_positions, 5))
        logger.info(
            "Dynamic sizing: No confidence/return/score data, defaulting to %d (max_positions=%d)",
            fallback,
            max_positions,
        )
        return fallback

    # Calculate quality score for each stock
    # Quality = weighted combination of confidence and predicted return percentile
    quality_scores = pd.Series(0.0, index=screened.index)

    if has_confidence:
        # Normalize confidence to 0-1 range (it's typically already 0-1)
        conf_normalized = screened["pred_confidence"].clip(0, 1)
        quality_scores += conf_normalized * 0.4  # 40% weight to confidence

    if has_return:
        # Normalize predicted return using percentile rank within screened set
        ret_pct = screened["pred_return"].rank(pct=True)
        quality_scores += ret_pct * 0.6  # 60% weight to return (more important)

    # Determine quality threshold based on model IC
    # High IC → lower threshold (be more aggressive)
    # Low/negative IC → higher threshold (be conservative)
    if model_ic is not None and model_ic > 0:
        # IC typically ranges 0.01-0.10 for good models
        # Scale threshold: IC=0.10 → threshold=0.4, IC=0.01 → threshold=0.7
        ic_factor = min(1.0, max(0.0, model_ic * 10))  # 0-1 based on IC
        quality_threshold = 0.7 - (ic_factor * 0.3)  # Range: 0.4-0.7
        logger.info("Dynamic sizing: Model IC=%.3f, quality threshold=%.2f", model_ic, quality_threshold)
    else:
        # Conservative default when no IC available
        quality_threshold = 0.6
        logger.info("Dynamic sizing: No model IC, using default threshold=%.2f", quality_threshold)

    # Apply hard thresholds first
    qualifying_mask = pd.Series(True, index=screened.index)

    if has_confidence:
        qualifying_mask &= (screened["pred_confidence"] >= min_confidence)

    if has_return:
        qualifying_mask &= (screened["pred_return"] >= min_pred_return)

    # Then apply quality score threshold
    qualifying_mask &= (quality_scores >= quality_threshold)

    qualifying_count = int(qualifying_mask.sum())

    # Portfolio size: at least 1 (always recommend something), at most max_positions
    dynamic_size = max(1, min(max_positions, qualifying_count))

    # Log details
    default_metric = pd.Series(1.0, index=screened.index, dtype=float)
    hard_threshold_count = (
        (screened.get("pred_confidence", default_metric) >= min_confidence)
        & (screened.get("pred_return", default_metric) >= min_pred_return)
    ).sum()

    logger.info(
        "Dynamic sizing: %d stocks pass hard thresholds (conf>=%.2f, ret>=%.1f%%), "
        "%d pass quality threshold (>=%.2f), final portfolio size: %d",
        hard_threshold_count, min_confidence, min_pred_return * 100,
        qualifying_count, quality_threshold, dynamic_size
    )

    return dynamic_size


def _lookup_frame_metric(
    frame: pd.DataFrame | None,
    ticker: str,
    column: str,
) -> float | None:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    t = str(ticker).strip()
    if not t:
        return None

    for key in (t, t.upper()):
        if key in frame.index:
            val = frame.loc[key, column]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if pd.isna(v):
                continue
            return v

    t_upper = t.upper()
    for i, idx_val in enumerate(frame.index):
        if str(idx_val).upper() != t_upper:
            continue
        val = frame.iloc[i][column]
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        if pd.isna(v):
            return None
        return v
    return None


def _lookup_metric_from_sources(
    ticker: str,
    column: str,
    sources: list[pd.DataFrame | None],
) -> float | None:
    for frame in sources:
        val = _lookup_frame_metric(frame, ticker, column)
        if val is not None:
            return val
    return None


def _extract_model_holdout_ic(model_metadata: dict[str, Any] | None) -> float | None:
    """Extract holdout IC from current or legacy model metadata layouts."""
    if not isinstance(model_metadata, dict):
        return None

    raw_ic = None
    reg_payload = model_metadata.get("regressor")
    if isinstance(reg_payload, dict):
        reg_holdout = reg_payload.get("holdout")
        if isinstance(reg_holdout, dict):
            raw_ic = reg_holdout.get("mean_ic")

    if raw_ic is None:
        legacy_holdout = model_metadata.get("holdout")
        if isinstance(legacy_holdout, dict):
            raw_ic = legacy_holdout.get("mean_ic")

    try:
        model_ic = float(raw_ic)
    except (TypeError, ValueError):
        return None
    if pd.isna(model_ic):
        return None
    return model_ic


def _check_runtime_budget(started_utc: datetime, cfg: Config, logger, stage: str) -> None:
    max_minutes = max(1, int(getattr(cfg, "max_daily_runtime_minutes", 12)))
    elapsed_minutes = (datetime.now(tz=timezone.utc) - started_utc).total_seconds() / 60.0
    if elapsed_minutes > max_minutes:
        raise TimeoutError(
            f"Runtime budget exceeded at stage '{stage}': "
            f"{elapsed_minutes:.1f}m > {max_minutes}m"
        )
    if elapsed_minutes > max_minutes * 0.8:
        logger.warning(
            "Runtime budget nearing limit at %s: %.1fm / %dm",
            stage, elapsed_minutes, max_minutes,
        )


def _series_distribution_stats(series: pd.Series | None) -> dict[str, Any]:
    if series is None:
        return {"n": 0}
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
    }


def _compute_ret_per_day_signal(
    features: pd.DataFrame,
    *,
    cfg: Config,
    logger,
    previous_run_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "enabled": False,
        "reason": "missing_columns",
        "shift_alert_triggered": False,
        "shift_alerts": [],
    }
    if features is None or features.empty:
        info["reason"] = "empty_features"
        return info
    if "pred_return" not in features.columns or "pred_peak_days" not in features.columns:
        return info

    pred_return = pd.to_numeric(features["pred_return"], errors="coerce")
    peak_raw = pd.to_numeric(features["pred_peak_days"], errors="coerce")
    if pred_return.notna().sum() <= 0 or peak_raw.notna().sum() <= 0:
        info["reason"] = "no_valid_pred_return_or_peak_days"
        return info

    min_days = float(max(0.1, getattr(cfg, "ret_per_day_min_peak_days", 1.0)))
    max_days = float(max(min_days, getattr(cfg, "ret_per_day_max_peak_days", 10.0)))
    smoothing_k = float(max(0.0, getattr(cfg, "ret_per_day_smoothing_k", 1.0)))

    peak_clamped = peak_raw.clip(lower=min_days, upper=max_days)
    denom = peak_clamped + smoothing_k
    ret_per_day = pred_return / denom
    ret_per_day = pd.to_numeric(ret_per_day, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)

    features["pred_peak_days_raw"] = peak_raw
    features["pred_peak_days"] = peak_clamped
    features["ret_per_day"] = ret_per_day

    clip_low = int((peak_raw < min_days).fillna(False).sum())
    clip_high = int((peak_raw > max_days).fillna(False).sum())
    valid_peak = int(peak_raw.notna().sum())
    clipped_total = clip_low + clip_high
    clip_share = float(clipped_total / valid_peak) if valid_peak > 0 else 0.0

    peak_raw_stats = _series_distribution_stats(peak_raw)
    peak_clamped_stats = _series_distribution_stats(peak_clamped)
    ret_stats = _series_distribution_stats(ret_per_day)

    info.update(
        {
            "enabled": True,
            "reason": "ok",
            "formula": "ret_per_day = pred_return / (clamp(pred_peak_days, min_days, max_days) + k)",
            "min_peak_days": min_days,
            "max_peak_days": max_days,
            "smoothing_k": smoothing_k,
            "peak_days_clip_low_count": clip_low,
            "peak_days_clip_high_count": clip_high,
            "peak_days_clip_total_count": clipped_total,
            "peak_days_clip_share": clip_share,
            "pred_peak_days_raw_stats": peak_raw_stats,
            "pred_peak_days_clamped_stats": peak_clamped_stats,
            "ret_per_day_stats": ret_stats,
            "shift_alert_triggered": False,
            "shift_alerts": [],
        }
    )

    logger.info(
        "ret_per_day guardrails: min_days=%.2f max_days=%.2f k=%.2f clipped=%d/%d (%.1f%%)",
        min_days,
        max_days,
        smoothing_k,
        clipped_total,
        valid_peak,
        clip_share * 100.0,
    )
    logger.info(
        "ret_per_day dist: mean=%.5f p10=%.5f p50=%.5f p90=%.5f max=%.5f n=%d",
        float(ret_stats.get("mean", float("nan"))),
        float(ret_stats.get("p10", float("nan"))),
        float(ret_stats.get("p50", float("nan"))),
        float(ret_stats.get("p90", float("nan"))),
        float(ret_stats.get("max", float("nan"))),
        int(ret_stats.get("n", 0)),
    )

    alert_enabled = bool(getattr(cfg, "ret_per_day_shift_alert_enabled", True))
    min_n = int(max(5, getattr(cfg, "ret_per_day_shift_alert_min_samples", 20)))
    if not alert_enabled:
        return info

    prev_stats = None
    if isinstance(previous_run_meta, dict):
        prev_signal = previous_run_meta.get("ret_per_day_signal")
        if isinstance(prev_signal, dict):
            candidate = prev_signal.get("ret_per_day_stats")
            if isinstance(candidate, dict):
                prev_stats = candidate

    info["previous_stats_available"] = bool(isinstance(prev_stats, dict))
    if not isinstance(prev_stats, dict):
        return info
    if int(ret_stats.get("n", 0)) < min_n or int(prev_stats.get("n", 0)) < min_n:
        info["shift_alert_reason"] = "insufficient_samples"
        return info

    metric_thresholds = {
        "mean": float(max(0.0, getattr(cfg, "ret_per_day_mean_shift_alert_pct", 0.50))),
        "p90": float(max(0.0, getattr(cfg, "ret_per_day_p90_shift_alert_pct", 0.50))),
        "std": float(max(0.0, getattr(cfg, "ret_per_day_std_shift_alert_pct", 0.75))),
    }
    alerts: list[dict[str, Any]] = []
    for metric, threshold in metric_thresholds.items():
        curr = ret_stats.get(metric)
        prev = prev_stats.get(metric)
        if curr is None or prev is None:
            continue
        try:
            curr_f = float(curr)
            prev_f = float(prev)
        except (TypeError, ValueError):
            continue
        if pd.isna(curr_f) or pd.isna(prev_f):
            continue
        denom = max(abs(prev_f), 1e-9)
        rel_shift = abs(curr_f - prev_f) / denom
        if rel_shift >= threshold:
            alerts.append(
                {
                    "metric": metric,
                    "previous": prev_f,
                    "current": curr_f,
                    "relative_shift": float(rel_shift),
                    "threshold": float(threshold),
                }
            )

    if alerts:
        info["shift_alert_triggered"] = True
        info["shift_alerts"] = alerts
        logger.warning("ret_per_day distribution shift alert(s): %s", alerts)
    return info


def _compute_effective_entry_thresholds(
    screened: pd.DataFrame,
    *,
    cfg: Config,
    logger,
) -> dict[str, Any]:
    """Compute effective entry thresholds with optional dynamic relaxation.

    Dynamic mode is relax-only: it never makes thresholds stricter than explicit
    config values, and always respects configured floors.
    """
    min_conf = getattr(cfg, "entry_min_confidence", None)
    min_pred = getattr(cfg, "entry_min_pred_return", None)
    conf_floor = float(getattr(cfg, "entry_min_confidence_floor", 0.35))
    conf_floor = float(max(0.0, min(1.0, conf_floor)))
    pred_floor = float(getattr(cfg, "entry_min_pred_return_floor", 0.0025))

    if min_conf is not None:
        min_conf = max(conf_floor, float(min_conf))
    if min_pred is not None:
        min_pred = max(pred_floor, float(min_pred))

    result: dict[str, Any] = {
        "min_confidence": min_conf,
        "min_pred_return": min_pred,
        "dynamic_applied": False,
        "dynamic_blocked_reasons": [],
        "stress_guard_triggered": False,
        "hold_only_recommended": False,
        "floors": {
            "min_confidence_floor": conf_floor,
            "min_pred_return_floor": pred_floor,
        },
    }

    if screened is None or screened.empty:
        return result

    # Stress guard: tighten thresholds under weak market conditions and optionally
    # escalate to HOLD_ONLY.
    stress_guard_enabled = bool(getattr(cfg, "entry_stress_guard_enabled", True))
    max_vol_stress = float(max(0.0, min(1.0, getattr(cfg, "entry_stress_max_vol_stress", 0.65))))
    min_breadth = float(max(0.0, min(1.0, getattr(cfg, "entry_stress_min_breadth", 0.45))))
    conf_tighten_add = float(max(0.0, getattr(cfg, "entry_stress_confidence_tighten_add", 0.05)))
    pred_tighten_add = float(max(0.0, getattr(cfg, "entry_stress_pred_return_tighten_add", 0.002)))
    hold_only_on_stress = bool(getattr(cfg, "entry_stress_hold_only_enabled", False))

    vol_regime = None
    if "market_vol_regime" in screened.columns:
        v = pd.to_numeric(screened["market_vol_regime"], errors="coerce").dropna()
        if not v.empty:
            vol_regime = float(v.iloc[0])
    vol_stress = None
    if vol_regime is not None and pd.notna(vol_regime):
        vol_stress = float(max(0.0, min(1.0, (float(vol_regime) - 1.0) / 0.8)))

    market_breadth = None
    if "market_breadth" in screened.columns:
        b = pd.to_numeric(screened["market_breadth"], errors="coerce").dropna()
        if not b.empty:
            market_breadth = float(b.iloc[0])

    high_vol_stress = bool(vol_stress is not None and vol_stress >= max_vol_stress)
    poor_breadth = bool(market_breadth is not None and market_breadth <= min_breadth)
    stress_triggered = bool(stress_guard_enabled and (high_vol_stress or poor_breadth))
    stress_reasons: list[str] = []
    if high_vol_stress:
        stress_reasons.append("vol_stress_high")
    if poor_breadth:
        stress_reasons.append("breadth_poor")

    result["stress_guard"] = {
        "enabled": stress_guard_enabled,
        "triggered": stress_triggered,
        "vol_regime": vol_regime,
        "vol_stress": vol_stress,
        "vol_stress_threshold": max_vol_stress,
        "market_breadth": market_breadth,
        "breadth_threshold": min_breadth,
        "reasons": stress_reasons,
        "hold_only_enabled": hold_only_on_stress,
    }
    result["stress_guard_triggered"] = stress_triggered

    if stress_triggered:
        if min_conf is None:
            result["min_confidence"] = conf_floor + conf_tighten_add
        else:
            result["min_confidence"] = max(conf_floor, float(min_conf) + conf_tighten_add)
        if min_pred is None:
            result["min_pred_return"] = pred_floor + pred_tighten_add
        else:
            result["min_pred_return"] = max(pred_floor, float(min_pred) + pred_tighten_add)
        result["stress_tightening_applied"] = True
        if hold_only_on_stress:
            result["hold_only_recommended"] = True
        logger.warning(
            "Entry stress guard triggered (%s): tightened thresholds to conf>=%.3f pred_return>=%.2f%%",
            ",".join(stress_reasons) if stress_reasons else "stress_signal",
            float(result["min_confidence"]) if result["min_confidence"] is not None else float("nan"),
            float(result["min_pred_return"]) * 100.0 if result["min_pred_return"] is not None else float("nan"),
        )

    dynamic_enabled = bool(getattr(cfg, "entry_dynamic_thresholds_enabled", True))
    if not dynamic_enabled:
        result["dynamic_blocked_reasons"].append("disabled")
        return result

    min_candidates = max(5, int(getattr(cfg, "entry_dynamic_min_candidates", 20)))
    if len(screened) < min_candidates:
        result["dynamic_blocked_reasons"].append("insufficient_candidates")
        return result

    if stress_triggered:
        result["dynamic_blocked_reasons"].append("stress_guard_triggered")
        return result

    dynamic_updates: dict[str, float] = {}
    separation: dict[str, Any] = {}
    min_conf_spread = float(max(0.0, getattr(cfg, "entry_dynamic_min_conf_top_decile_spread", 0.03)))
    min_pred_spread = float(max(0.0, getattr(cfg, "entry_dynamic_min_pred_top_decile_spread", 0.002)))

    if "pred_confidence" in screened.columns:
        conf = pd.to_numeric(screened["pred_confidence"], errors="coerce").dropna()
        if len(conf) >= min_candidates:
            conf_top_dec = float(conf.quantile(0.90))
            conf_med = float(conf.quantile(0.50))
            conf_spread = float(conf_top_dec - conf_med)
            separation["conf_top_decile"] = conf_top_dec
            separation["conf_median"] = conf_med
            separation["conf_top_decile_spread"] = conf_spread
            separation["conf_min_required_spread"] = min_conf_spread
            if conf_spread >= min_conf_spread:
                conf_pct = float(getattr(cfg, "entry_confidence_percentile", 0.35))
                conf_pct = float(max(0.0, min(1.0, conf_pct)))
                dyn_conf = float(conf.quantile(conf_pct))
                if min_conf is None:
                    eff_conf = max(conf_floor, dyn_conf)
                else:
                    eff_conf = max(conf_floor, min(float(min_conf), dyn_conf))
                if min_conf is None or eff_conf < float(min_conf) - 1e-12:
                    dynamic_updates["min_confidence"] = float(eff_conf)
                    result["min_confidence"] = float(eff_conf)
            else:
                result["dynamic_blocked_reasons"].append("confidence_spread_too_low")

    if "pred_return" in screened.columns:
        pred = pd.to_numeric(screened["pred_return"], errors="coerce").dropna()
        if len(pred) >= min_candidates:
            pred_top_dec = float(pred.quantile(0.90))
            pred_med = float(pred.quantile(0.50))
            pred_spread = float(pred_top_dec - pred_med)
            separation["pred_top_decile"] = pred_top_dec
            separation["pred_median"] = pred_med
            separation["pred_top_decile_spread"] = pred_spread
            separation["pred_min_required_spread"] = min_pred_spread
            if pred_spread >= min_pred_spread:
                pred_pct = float(getattr(cfg, "entry_pred_return_percentile", 0.60))
                pred_pct = float(max(0.0, min(1.0, pred_pct)))
                dyn_pred = float(pred.quantile(pred_pct))
                if min_pred is None:
                    eff_pred = max(pred_floor, dyn_pred)
                else:
                    eff_pred = max(pred_floor, min(float(min_pred), dyn_pred))
                if min_pred is None or eff_pred < float(min_pred) - 1e-12:
                    dynamic_updates["min_pred_return"] = float(eff_pred)
                    result["min_pred_return"] = float(eff_pred)
            else:
                result["dynamic_blocked_reasons"].append("pred_return_spread_too_low")

    if separation:
        result["separation"] = separation

    if dynamic_updates:
        result["dynamic_applied"] = True
        result["dynamic_updates"] = dynamic_updates
        logger.info(
            "Dynamic entry thresholds applied: conf>=%.3f pred_return>=%.2f%%",
            float(result["min_confidence"]) if result["min_confidence"] is not None else float("nan"),
            float(result["min_pred_return"]) * 100.0 if result["min_pred_return"] is not None else float("nan"),
        )
    elif not result["dynamic_blocked_reasons"]:
        result["dynamic_blocked_reasons"].append("no_relaxation_needed")
    return result


def _infer_instrument_type_row(row: pd.Series) -> str:
    """Classify ticker as EQUITY or FUND using fundamentals metadata + fallbacks."""
    quote_type = str(row.get("quote_type", "") or "").strip().lower()
    fund_family = str(row.get("fund_family", "") or "").strip().lower()
    fund_category = str(row.get("fund_category", "") or "").strip().lower()
    sector = str(row.get("sector", "") or "").strip()
    industry = str(row.get("industry", "") or "").strip()
    log_mcap = pd.to_numeric(row.get("log_market_cap"), errors="coerce")

    fund_score = 0
    equity_score = 0

    if quote_type == "equity":
        equity_score += 3
    if any(k in quote_type for k in ("etf", "fund", "mutual")):
        fund_score += 3
    if fund_family:
        fund_score += 1
    if fund_category:
        fund_score += 1
    if sector:
        equity_score += 1
    if industry:
        equity_score += 1
    if pd.notna(log_mcap):
        equity_score += 1
    if pd.isna(log_mcap) and not sector and not industry:
        fund_score += 1

    return "FUND" if fund_score > equity_score else "EQUITY"


def _infer_instrument_types(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=object)
    out = frame.apply(_infer_instrument_type_row, axis=1)
    return out.astype(str)


def _apply_instrument_sleeve_constraints(
    target_weights: pd.DataFrame,
    *,
    screened: pd.DataFrame,
    cfg: Config,
    logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cap fund sleeve and enforce minimum equity sleeve without changing gross exposure."""
    info: dict[str, Any] = {"enabled": bool(getattr(cfg, "instrument_sleeve_constraints_enabled", True))}
    if target_weights.empty or "weight" not in target_weights.columns:
        info["applied"] = False
        info["reason"] = "empty_target_weights"
        return target_weights, info
    if not info["enabled"]:
        info["applied"] = False
        info["reason"] = "disabled"
        return target_weights, info

    out = target_weights.copy()
    w = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    invested = float(w.sum())
    if invested <= 0:
        info["applied"] = False
        info["reason"] = "zero_invested_weight"
        return out, info

    instrument_frame = screened.reindex(out.index) if screened is not None and not screened.empty else out
    types = _infer_instrument_types(instrument_frame).reindex(out.index).fillna("EQUITY")
    fund_mask = types == "FUND"
    equity_mask = ~fund_mask

    if not bool(fund_mask.any()) or not bool(equity_mask.any()):
        info["applied"] = False
        info["reason"] = "single_sleeve_only"
        info["fund_count"] = int(fund_mask.sum())
        info["equity_count"] = int(equity_mask.sum())
        return out, info

    max_fund = float(max(0.0, min(1.0, float(getattr(cfg, "instrument_fund_max_weight", 0.35)))))
    min_equity = float(max(0.0, min(1.0, float(getattr(cfg, "instrument_equity_min_weight", 0.50)))))
    eff_max_fund = min(max_fund, invested)
    eff_min_equity = min(min_equity, invested)

    fund_before = float(w[fund_mask].sum())
    equity_before = float(w[equity_mask].sum())
    shift_for_fund_cap = max(0.0, fund_before - eff_max_fund)
    shift_for_equity_floor = max(0.0, eff_min_equity - equity_before)
    shift = min(fund_before, max(shift_for_fund_cap, shift_for_equity_floor))

    if shift <= 1e-12:
        info.update(
            {
                "applied": False,
                "reason": "already_within_bounds",
                "fund_weight_before": fund_before,
                "equity_weight_before": equity_before,
                "fund_weight_after": fund_before,
                "equity_weight_after": equity_before,
                "invested_weight": invested,
                "max_fund_weight": eff_max_fund,
                "min_equity_weight": eff_min_equity,
                "fund_count": int(fund_mask.sum()),
                "equity_count": int(equity_mask.sum()),
            }
        )
        return out, info

    fund_weights = w[fund_mask]
    equity_weights = w[equity_mask]
    fund_total = float(fund_weights.sum())
    equity_total = float(equity_weights.sum())
    if fund_total <= 0:
        info["applied"] = False
        info["reason"] = "no_fund_weight"
        return out, info

    # Remove from fund sleeve proportionally.
    reduce_ratio = shift / fund_total
    fund_new = fund_weights * max(0.0, (1.0 - reduce_ratio))
    out.loc[fund_mask, "weight"] = fund_new

    # Add to equity sleeve proportionally (or evenly when equity sleeve is zero).
    if equity_total > 0:
        equity_new = equity_weights + (equity_weights / equity_total) * shift
    else:
        n_eq = int(equity_mask.sum())
        if n_eq <= 0:
            info["applied"] = False
            info["reason"] = "no_equity_receivers"
            return target_weights, info
        equity_new = pd.Series(shift / n_eq, index=equity_weights.index, dtype=float)
    out.loc[equity_mask, "weight"] = equity_new

    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    after_sum = float(out["weight"].sum())
    if after_sum > 0 and abs(after_sum - invested) > 1e-8:
        out["weight"] = out["weight"] * (invested / after_sum)

    fund_after = float(out.loc[fund_mask, "weight"].sum())
    equity_after = float(out.loc[equity_mask, "weight"].sum())
    info.update(
        {
            "applied": True,
            "fund_weight_before": fund_before,
            "equity_weight_before": equity_before,
            "fund_weight_after": fund_after,
            "equity_weight_after": equity_after,
            "invested_weight": invested,
            "max_fund_weight": eff_max_fund,
            "min_equity_weight": eff_min_equity,
            "fund_count": int(fund_mask.sum()),
            "equity_count": int(equity_mask.sum()),
            "shift_weight": shift,
        }
    )
    logger.info(
        "Instrument sleeve rebalance: fund %.1f%% -> %.1f%%, equity %.1f%% -> %.1f%% (shift=%.1f%%)",
        fund_before * 100.0,
        fund_after * 100.0,
        equity_before * 100.0,
        equity_after * 100.0,
        shift * 100.0,
    )
    return out, info


def _validate_feature_parity(
    features_df: pd.DataFrame,
    selected_features: list[str] | None,
    *,
    strict: bool,
    logger,
) -> list[str]:
    if not selected_features:
        return []
    missing = [c for c in selected_features if c not in features_df.columns]
    if missing:
        msg = (
            f"Feature schema mismatch: missing {len(missing)} selected features "
            f"(examples: {missing[:8]})"
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s. Continuing with NaN-filled fallback.", msg)
    return missing


def _apply_uncertainty_weighting(
    target_weights: pd.DataFrame,
    screened: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    if target_weights.empty or "pred_uncertainty" not in screened.columns:
        return target_weights
    out = target_weights.copy()
    uncertainty = screened["pred_uncertainty"].reindex(out.index)
    if uncertainty.isna().all():
        return out
    # Higher uncertainty -> smaller weight; bounded and smooth.
    scalar = 1.0 / (1.0 + uncertainty.fillna(uncertainty.median()).clip(lower=0.0))
    out["weight"] = out["weight"] * scalar
    total = float(out["weight"].sum())
    if total > 0:
        out["weight"] = out["weight"] / total
    logger.info(
        "Applied uncertainty weighting: uncertainty range [%.4f, %.4f]",
        float(uncertainty.min(skipna=True)),
        float(uncertainty.max(skipna=True)),
    )
    return out.sort_values("weight", ascending=False)


def _compute_exposure_metrics(
    weights: pd.DataFrame,
    *,
    target_gross: float = 1.0,
) -> dict[str, float]:
    if weights is None or weights.empty or "weight" not in weights.columns:
        return {
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "cash_weight": float(target_gross),
        }
    w = pd.to_numeric(weights["weight"], errors="coerce").fillna(0.0)
    gross = float(w.abs().sum())
    net = float(w.sum())
    cash = float(target_gross) - gross
    return {
        "gross_exposure": gross,
        "net_exposure": net,
        "cash_weight": cash,
    }


def _weights_to_series(weights: pd.DataFrame | None) -> pd.Series:
    if weights is None or weights.empty or "weight" not in weights.columns:
        return pd.Series(dtype=float)
    out = pd.to_numeric(weights["weight"], errors="coerce").fillna(0.0)
    out.index = out.index.astype(str)
    if out.index.has_duplicates:
        out = out.groupby(level=0).sum()
    return out.astype(float).sort_index()


def _compute_weight_distance_metrics(
    reference_weights: pd.DataFrame | None,
    final_weights: pd.DataFrame | None,
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    ref = _weights_to_series(reference_weights)
    fin = _weights_to_series(final_weights)
    idx = ref.index.union(fin.index)
    if len(idx) == 0:
        return {
            "l1_distance": 0.0,
            "l2_distance": 0.0,
            "max_abs_diff": 0.0,
            "changed_count": 0,
            "dropped_count": 0,
            "entered_count": 0,
            "dropped_tickers": [],
            "entered_tickers": [],
            "top_abs_drift": [],
        }

    ref_a = ref.reindex(idx).fillna(0.0).astype(float)
    fin_a = fin.reindex(idx).fillna(0.0).astype(float)
    delta = fin_a - ref_a
    abs_delta = delta.abs()

    changed = abs_delta[abs_delta > 1e-12]
    dropped = sorted(idx[(ref_a > 1e-12) & (fin_a <= 1e-12)].astype(str).tolist())
    entered = sorted(idx[(ref_a <= 1e-12) & (fin_a > 1e-12)].astype(str).tolist())

    top = (
        pd.DataFrame(
            {
                "ticker": idx.astype(str),
                "reference_weight": ref_a.values,
                "final_weight": fin_a.values,
                "abs_diff": abs_delta.values,
            }
        )
        .sort_values("abs_diff", ascending=False)
        .head(max(1, int(top_n)))
    )
    top_records = []
    for rec in top.to_dict(orient="records"):
        if float(rec.get("abs_diff", 0.0)) <= 1e-12:
            continue
        top_records.append(
            {
                "ticker": str(rec["ticker"]),
                "reference_weight": float(rec["reference_weight"]),
                "final_weight": float(rec["final_weight"]),
                "abs_diff": float(rec["abs_diff"]),
            }
        )

    return {
        "l1_distance": float(abs_delta.sum()),
        "l2_distance": float((delta.pow(2).sum()) ** 0.5),
        "max_abs_diff": float(abs_delta.max()) if len(abs_delta) else 0.0,
        "changed_count": int(len(changed)),
        "dropped_count": int(len(dropped)),
        "entered_count": int(len(entered)),
        "dropped_tickers": dropped,
        "entered_tickers": entered,
        "top_abs_drift": top_records,
    }


def _enforce_exposure_policy(
    weights: pd.DataFrame,
    *,
    exposure_policy: str,
    target_gross_exposure: float,
    allow_leverage: bool,
    logger,
    context: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = weights.copy()
    policy_raw = str(exposure_policy or "allow_cash_no_upscale").strip().lower()
    policy = policy_raw if policy_raw in {"allow_cash_no_upscale", "normalize_to_target_gross"} else "allow_cash_no_upscale"
    configured_target_gross = max(0.0, float(target_gross_exposure))
    effective_target_gross = configured_target_gross if bool(allow_leverage) else min(1.0, configured_target_gross)
    hard_gross_cap = effective_target_gross

    before = _compute_exposure_metrics(out, target_gross=effective_target_gross)
    changed = False
    reason = "none"

    if out.empty or "weight" not in out.columns:
        after = _compute_exposure_metrics(out, target_gross=effective_target_gross)
        return out, {
            "policy": policy,
            "target_gross_exposure": configured_target_gross,
            "effective_target_gross_exposure": effective_target_gross,
            "allow_leverage": bool(allow_leverage),
            "changed": changed,
            "reason": reason,
            "before": before,
            "after": after,
        }

    gross_before = float(before["gross_exposure"])
    if policy == "normalize_to_target_gross":
        if gross_before > 0 and abs(gross_before - effective_target_gross) > 1e-12:
            out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0) * (effective_target_gross / gross_before)
            changed = True
            reason = "scaled_to_target_gross"
    else:
        if gross_before > hard_gross_cap + 1e-12:
            out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0) * (hard_gross_cap / gross_before)
            changed = True
            reason = "downscaled_infeasible_gross"
        else:
            reason = "cash_allowed_no_upscale"

    after = _compute_exposure_metrics(out, target_gross=effective_target_gross)
    logger.info(
        "Exposure policy (%s @ %s, target=%.4f effective=%.4f): gross %.4f -> %.4f, net %.4f -> %.4f, cash %.4f -> %.4f",
        policy,
        context,
        configured_target_gross,
        effective_target_gross,
        float(before["gross_exposure"]),
        float(after["gross_exposure"]),
        float(before["net_exposure"]),
        float(after["net_exposure"]),
        float(before["cash_weight"]),
        float(after["cash_weight"]),
    )
    return out, {
        "policy": policy,
        "target_gross_exposure": configured_target_gross,
        "effective_target_gross_exposure": effective_target_gross,
        "allow_leverage": bool(allow_leverage),
        "changed": changed,
        "reason": reason,
        "before": before,
        "after": after,
    }

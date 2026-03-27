from __future__ import annotations

from typing import Any

import pandas as pd

from stock_screener.pipeline.helpers import _lookup_frame_metric, _enforce_exposure_policy


def _apply_rebalance_controls(
    target_weights: pd.DataFrame,
    *,
    state,
    screened: pd.DataFrame,
    prices_cad: pd.Series,
    market_vol_regime: float | None,
    min_rebalance_weight_delta: float,
    min_trade_notional_cad: float,
    turnover_penalty_bps: float,
    dynamic_band_enabled: bool,
    uncertainty_weight: float,
    liquidity_weight: float,
    vol_regime_weight: float,
    band_mult_min: float,
    band_mult_max: float,
    logger,
    exposure_policy: str = "allow_cash_no_upscale",
    target_gross_exposure: float = 1.0,
    allow_leverage: bool = False,
    apply_turnover_shrinkage: bool = True,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "input_weight_count": int(len(target_weights)) if target_weights is not None else 0,
        "output_weight_count": 0,
        "path": "unknown",
        "equity_cad": 0.0,
        "min_rebalance_weight_delta": 0.0,
        "min_trade_notional_cad": 0.0,
        "dynamic_band_enabled": bool(dynamic_band_enabled),
        "apply_turnover_shrinkage": bool(apply_turnover_shrinkage),
        "gate_reason_counts": {},
        "dropped_tickers": [],
        "dropped_by_notional_gate": [],
        "hysteresis_kept_count": 0,
        "fallback_used": False,
    }

    def _increment_reason(reason_code: str) -> None:
        code = str(reason_code or "unknown")
        counts = diagnostics["gate_reason_counts"]
        counts[code] = int(counts.get(code, 0)) + 1

    def _record_drop(
        *,
        ticker: str,
        reason_code: str,
        target_weight: float,
        current_weight: float,
        delta_weight: float,
        trade_notional_cad: float | None = None,
        delta_threshold: float | None = None,
        notional_threshold_cad: float | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "ticker": str(ticker),
            "reason_code": str(reason_code),
            "target_weight": float(target_weight),
            "current_weight": float(current_weight),
            "delta_weight": float(delta_weight),
        }
        if trade_notional_cad is not None:
            event["trade_notional_cad"] = float(trade_notional_cad)
        if delta_threshold is not None:
            event["delta_threshold"] = float(delta_threshold)
        if notional_threshold_cad is not None:
            event["notional_threshold_cad"] = float(notional_threshold_cad)
        diagnostics["dropped_tickers"].append(event)
        if "notional" in str(reason_code):
            diagnostics["dropped_by_notional_gate"].append(event)

    def _finalize(
        out: pd.DataFrame,
        *,
        path: str,
        exposure_info: dict[str, Any] | None = None,
        fallback_used: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
        out_sorted = out.sort_values("weight", ascending=False) if isinstance(out, pd.DataFrame) and "weight" in out.columns else out
        diagnostics["path"] = str(path)
        diagnostics["fallback_used"] = bool(fallback_used)
        diagnostics["output_weight_count"] = int(len(out_sorted)) if isinstance(out_sorted, pd.DataFrame) else 0
        diagnostics["hysteresis_kept_count"] = int(diagnostics.get("hysteresis_kept_count", 0))
        if isinstance(exposure_info, dict):
            diagnostics["exposure_policy"] = exposure_info

        # Deterministic ordering + de-duplication for audit logs.
        dropped_events = diagnostics.get("dropped_tickers", [])
        if isinstance(dropped_events, list) and dropped_events:
            uniq: dict[tuple[str, str], dict[str, Any]] = {}
            for evt in dropped_events:
                if not isinstance(evt, dict):
                    continue
                ticker = str(evt.get("ticker", "")).upper()
                reason = str(evt.get("reason_code", "unknown"))
                key = (ticker, reason)
                prev = uniq.get(key)
                if prev is None or float(evt.get("target_weight", 0.0) or 0.0) > float(prev.get("target_weight", 0.0) or 0.0):
                    evt_copy = dict(evt)
                    evt_copy["ticker"] = ticker
                    uniq[key] = evt_copy
            ordered = [uniq[k] for k in sorted(uniq.keys())]
            diagnostics["dropped_tickers"] = ordered
            diagnostics["dropped_by_notional_gate"] = [
                e for e in ordered if "notional" in str(e.get("reason_code", ""))
            ]
        else:
            diagnostics["dropped_tickers"] = []
            diagnostics["dropped_by_notional_gate"] = []

        if return_diagnostics:
            return out_sorted, diagnostics
        return out_sorted

    if target_weights.empty:
        return _finalize(target_weights, path="empty_target_weights")

    min_delta = max(0.0, float(min_rebalance_weight_delta))
    min_notional = max(1.0, float(min_trade_notional_cad))
    penalty_bps = max(0.0, float(turnover_penalty_bps))
    apply_turnover_shrink = bool(apply_turnover_shrinkage)
    dyn_enabled = bool(dynamic_band_enabled)
    dyn_u_w = max(0.0, float(uncertainty_weight))
    dyn_l_w = max(0.0, float(liquidity_weight))
    dyn_v_w = max(0.0, float(vol_regime_weight))
    dyn_min = max(0.2, float(band_mult_min))
    dyn_max = max(dyn_min, float(band_mult_max))
    diagnostics["min_rebalance_weight_delta"] = float(min_delta)
    diagnostics["min_trade_notional_cad"] = float(min_notional)

    if penalty_bps > 0 and not apply_turnover_shrink:
        logger.info(
            "Rebalance controls: turnover shrinkage disabled; keeping hysteresis/notional guards only.",
        )

    open_values: dict[str, float] = {}
    open_total = 0.0
    for p in state.positions:
        if p.status != "OPEN" or not p.ticker or p.shares <= 0:
            continue
        px = float(prices_cad.get(p.ticker, float("nan")))
        if pd.isna(px) or px <= 0:
            continue
        val = float(px) * float(p.shares)
        open_values[p.ticker] = val
        open_total += val

    equity = float(state.cash_cad) + float(open_total)
    diagnostics["equity_cad"] = float(equity)
    if equity <= 0:
        return _finalize(target_weights, path="non_positive_equity")

    current_weights = {t: v / equity for t, v in open_values.items() if v > 0}

    # Cold start: no open holdings means there is no turnover to suppress.
    # Keep only entries that clear minimum trade notional.
    if not current_weights:
        cold = target_weights.copy()
        if "weight" not in cold.columns:
            return _finalize(cold, path="cold_start_missing_weight_column")
        cold = cold[cold["weight"].fillna(0.0) > 0.0].copy()
        if cold.empty:
            return _finalize(cold, path="cold_start_no_positive_targets")
        cold_before_notional = set(cold.index.astype(str))
        cold = cold[(cold["weight"].astype(float) * equity) >= min_notional].copy()
        dropped_cold_notional = sorted(cold_before_notional - set(cold.index.astype(str)))
        for t in dropped_cold_notional:
            tw = _lookup_frame_metric(target_weights, t, "weight")
            tw = float(tw) if tw is not None else 0.0
            _record_drop(
                ticker=t,
                reason_code="cold_start_min_notional",
                target_weight=tw,
                current_weight=0.0,
                delta_weight=tw,
                trade_notional_cad=tw * equity,
                delta_threshold=min_delta,
                notional_threshold_cad=min_notional,
            )
        if cold.empty:
            top = target_weights.sort_values("weight", ascending=False).head(1).copy()
            if top.empty:
                return _finalize(top, path="cold_start_seed_missing")
            top_w = float(top["weight"].iloc[0])
            top_notional = top_w * equity
            if top_w <= 0 or top_notional < 1.0:
                return _finalize(target_weights.iloc[0:0].copy(), path="cold_start_seed_below_minimum")
            if top_notional < min_notional:
                logger.warning(
                    "Rebalance cold start: top allocation %.2f CAD is below min trade notional %.2f; "
                    "keeping a seed target to avoid zero-exposure lockout.",
                    top_notional,
                    min_notional,
                )
            logger.info("Rebalance cold start: keeping top position to avoid empty portfolio.")
            top, exposure_info = _enforce_exposure_policy(
                top,
                exposure_policy=exposure_policy,
                target_gross_exposure=target_gross_exposure,
                allow_leverage=allow_leverage,
                logger=logger,
                context="rebalance_cold_start_seed",
            )
            _increment_reason("cold_start_seed_kept")
            return _finalize(top, path="cold_start_seed", exposure_info=exposure_info)
        cold, exposure_info = _enforce_exposure_policy(
            cold,
            exposure_policy=exposure_policy,
            target_gross_exposure=target_gross_exposure,
            allow_leverage=allow_leverage,
            logger=logger,
            context="rebalance_cold_start",
        )
        logger.info(
            "Rebalance cold start: bypassed hysteresis, kept %d entries (equity=%.2f, min_notional=%.2f)",
            len(cold),
            equity,
            min_notional,
        )
        _increment_reason("cold_start_targets_kept")
        return _finalize(cold, path="cold_start", exposure_info=exposure_info)

    adjusted = target_weights.copy()
    effective_weights: dict[str, float] = {}
    skipped_small = 0
    dyn_multipliers: list[float] = []

    ticker_unct = pd.Series(0.5, index=adjusted.index.astype(str), dtype=float)
    ticker_illiq = pd.Series(0.5, index=adjusted.index.astype(str), dtype=float)
    if dyn_enabled and screened is not None and not screened.empty:
        idx = pd.Index(adjusted.index.astype(str))
        if "pred_uncertainty" in screened.columns:
            u = pd.to_numeric(screened["pred_uncertainty"], errors="coerce")
            u_pct = u.rank(pct=True, na_option="keep").reindex(idx).fillna(0.5).clip(0.0, 1.0)
            ticker_unct = u_pct
        if "avg_dollar_volume_cad" in screened.columns:
            adv = pd.to_numeric(screened["avg_dollar_volume_cad"], errors="coerce")
            liq_pct = adv.rank(pct=True, na_option="keep").reindex(idx).fillna(0.5).clip(0.0, 1.0)
            ticker_illiq = (1.0 - liq_pct).clip(0.0, 1.0)

    vol_stress = 0.0
    if dyn_enabled and market_vol_regime is not None and pd.notna(market_vol_regime):
        vol_stress = float(max(0.0, min(1.0, (float(market_vol_regime) - 1.0) / 0.8)))

    all_tickers = sorted(
        {str(t) for t in adjusted.index.astype(str)}.union(str(t) for t in current_weights.keys())
    )
    for t in all_tickers:
        tgt_w = float(adjusted.loc[t, "weight"]) if t in adjusted.index and "weight" in adjusted.columns else 0.0
        cur_w = float(current_weights.get(t, 0.0))
        delta = abs(tgt_w - cur_w)
        dyn_mult = 1.0
        if dyn_enabled:
            u_s = float(ticker_unct.get(t, 0.5))
            l_s = float(ticker_illiq.get(t, 0.5))
            dyn_mult = 1.0 + (dyn_u_w * u_s) + (dyn_l_w * l_s) + (dyn_v_w * vol_stress)
            dyn_mult = float(max(dyn_min, min(dyn_max, dyn_mult)))
        dyn_multipliers.append(dyn_mult)

        delta_threshold = min_delta * dyn_mult
        notional_threshold = min_notional * dyn_mult
        trade_notional = delta * equity

        reason_code = "target_applied"
        if delta < delta_threshold:
            effective = cur_w
            skipped_small += 1
            reason_code = "hysteresis_delta"
        elif trade_notional < notional_threshold:
            effective = cur_w
            skipped_small += 1
            reason_code = "hysteresis_notional"
        else:
            effective = tgt_w
            if apply_turnover_shrink and penalty_bps > 0:
                # Transaction-cost-aware shrinkage of turnover-heavy moves.
                turnover_penalty = penalty_bps * dyn_mult * 1e-4 * delta
                effective = max(0.0, tgt_w - turnover_penalty)
                reason_code = "turnover_shrinkage"

        # Avoid initiating tiny new positions.
        if cur_w <= 0 and effective > 0 and effective * equity < notional_threshold:
            reason_code = "entry_notional_blocked"
            effective = 0.0
        _increment_reason(reason_code)

        if tgt_w > 1e-12 and cur_w <= 1e-12 and effective <= 1e-12:
            _record_drop(
                ticker=t,
                reason_code=reason_code,
                target_weight=tgt_w,
                current_weight=cur_w,
                delta_weight=delta,
                trade_notional_cad=trade_notional,
                delta_threshold=delta_threshold,
                notional_threshold_cad=notional_threshold,
            )
        effective_weights[t] = effective

    if skipped_small > 0:
        if dyn_enabled and dyn_multipliers:
            logger.info(
                "Rebalance hysteresis kept %d small changes (dynamic band x%.2f avg, vol_regime=%s)",
                skipped_small,
                float(pd.Series(dyn_multipliers, dtype=float).mean()),
                f"{float(market_vol_regime):.2f}" if market_vol_regime is not None and pd.notna(market_vol_regime) else "n/a",
            )
        else:
            logger.info(
                "Rebalance hysteresis kept %d small-delta/notional changes",
                skipped_small,
            )
    diagnostics["hysteresis_kept_count"] = int(skipped_small)

    for t, w in effective_weights.items():
        if t not in adjusted.index:
            adjusted.loc[t] = pd.NA
        adjusted.loc[t, "weight"] = w

    adjusted = adjusted[adjusted["weight"].fillna(0.0) > 0.0].copy()
    if adjusted.empty:
        # Safety fallback: do not allow hysteresis to wipe all exposure.
        fallback = target_weights.copy()
        if "weight" in fallback.columns:
            fallback = fallback[fallback["weight"].fillna(0.0) > 0.0].copy()
            fallback_before_notional = set(fallback.index.astype(str))
            fallback = fallback[(fallback["weight"].astype(float) * equity) >= min_notional].copy()
            dropped_fallback_notional = sorted(fallback_before_notional - set(fallback.index.astype(str)))
            for t in dropped_fallback_notional:
                tw = _lookup_frame_metric(target_weights, t, "weight")
                tw = float(tw) if tw is not None else 0.0
                _record_drop(
                    ticker=t,
                    reason_code="fallback_min_notional",
                    target_weight=tw,
                    current_weight=float(current_weights.get(t, 0.0)),
                    delta_weight=abs(tw - float(current_weights.get(t, 0.0))),
                    trade_notional_cad=abs(tw - float(current_weights.get(t, 0.0))) * equity,
                    delta_threshold=min_delta,
                    notional_threshold_cad=min_notional,
                )
            if not fallback.empty:
                logger.warning(
                    "Rebalance controls removed all targets; falling back to %d feasible target entries",
                    len(fallback),
                )
                fallback, exposure_info = _enforce_exposure_policy(
                    fallback,
                    exposure_policy=exposure_policy,
                    target_gross_exposure=target_gross_exposure,
                    allow_leverage=allow_leverage,
                    logger=logger,
                    context="rebalance_fallback",
                )
                _increment_reason("fallback_used")
                return _finalize(
                    fallback,
                    path="fallback_targets",
                    exposure_info=exposure_info,
                    fallback_used=True,
                )

    adjusted, exposure_info = _enforce_exposure_policy(
        adjusted,
        exposure_policy=exposure_policy,
        target_gross_exposure=target_gross_exposure,
        allow_leverage=allow_leverage,
        logger=logger,
        context="rebalance_final",
    )
    return _finalize(adjusted, path="standard", exposure_info=exposure_info)

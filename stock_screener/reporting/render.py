from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt_money(x: float) -> str:
    try:
        v = float(x)
        if v != v:
            return "N/A"
        return f"{v:,.2f}"
    except Exception:
        return "N/A"


def _fmt_pct(x: float) -> str:
    try:
        v = float(x)
        if v != v:  # NaN
            return "N/A"
        return f"{v * 100.0:+.2f}%"
    except Exception:
        return "N/A"


def _fmt_num(x: float) -> str:
    try:
        v = float(x)
        if v != v:
            return "N/A"
        return f"{v:,.3f}"
    except Exception:
        return "N/A"


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_reports(
    reports_dir: Path,
    run_meta: dict[str, Any],
    universe_meta: dict[str, Any],
    screened: pd.DataFrame,
    weights: pd.DataFrame,
    trade_actions: list[Any] | None,
    logger,
    *,
    target_weights: pd.DataFrame | None = None,
    portfolio_pnl_history: list[dict[str, Any]] | None = None,
    fx_usdcad_rate: float | None = None,
    total_processed: int | None = None,
) -> None:
    """Write daily email HTML + text report + weights CSV to reports_dir."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    # CSV of portfolio weights
    csv_cols = [
        "weight",
        "actual_weight",
        "target_weight",
        "shares",
        "position_value_cad",
        "score",
        "last_close_cad",
        "ret_60d",
        "ret_120d",
        "vol_60d_ann",
        "avg_dollar_volume_cad",
        "rsi_14",
        "ma20_ratio",
        "is_tsx",
        "last_date",
    ]
    weights_out = weights.copy()
    for c in csv_cols:
        if c not in weights_out.columns:
            weights_out[c] = pd.NA
    weights_out[csv_cols].to_csv(reports_dir / "portfolio_weights.csv", index=True)

    # Text report
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("DAILY STOCK SCREENER + RISK PARITY PORTFOLIO (CAD BASE)")
    lines.append("=" * 78)
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("UNIVERSE")
    lines.append("-" * 78)
    lines.append(f"US meta:  {universe_meta.get('us', {})}")
    lines.append(f"TSX meta: {universe_meta.get('tsx', {})}")
    lines.append(f"Total requested: {universe_meta.get('total_requested')}")
    if total_processed is not None:
        lines.append(f"Number of tickers scanned: {total_processed:,}")
    lines.append(f"Top screened: {len(screened):,} tickers")
    lines.append("")

    def _to_float(x: Any) -> float | None:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def _action_value(action_obj: Any, field: str, default: Any = None) -> Any:
        if isinstance(action_obj, dict):
            val = action_obj.get(field, default)
        else:
            val = getattr(action_obj, field, default)
        return default if val is None else val

    def _fmt_ic_summary(summary: dict[str, Any] | None) -> str:
        if not summary:
            return "N/A"
        mean_ic = _fmt_num(summary.get("mean_ic"))
        std_ic = _fmt_num(summary.get("std_ic"))
        ic_ir = _fmt_num(summary.get("ic_ir"))
        n_days = summary.get("n_days", "N/A")
        return f"mean_ic={mean_ic} std_ic={std_ic} ic_ir={ic_ir} n_days={n_days}"

    model_meta = run_meta.get("model", {})
    model_metrics = model_meta.get("metadata") if isinstance(model_meta, dict) else None
    if model_metrics:
        lines.append("MODEL VALIDATION (Holdout IC)")
        lines.append("-" * 78)
        reg_summary = model_metrics.get("regressor", {}).get("holdout")
        lines.append(f"Regressor: {_fmt_ic_summary(reg_summary)}")
        lines.append("")

    # LLM Agent Analysis (text report)
    _llm_text = run_meta.get("llm_agent") if isinstance(run_meta, dict) else None
    if isinstance(_llm_text, dict) and not _llm_text.get("decisions") and _llm_text.get("status") != "disabled":
        _s = _llm_text.get("status", "unknown")
        _r = _llm_text.get("reason", "")
        lines.append(f"LLM AGENT: {_s.upper()}{(' — ' + _r) if _r else ''}")
        lines.append("")
    if isinstance(_llm_text, dict) and _llm_text.get("decisions"):
        lines.append("LLM AGENT ANALYSIS")
        lines.append("-" * 78)
        for _t, _info in _llm_text["decisions"].items():
            if not isinstance(_info, dict):
                continue
            lines.append(f"{_t}: {_info.get('rating', 'N/A')} (score={_info.get('score', 0):+.1f})")
            _reason = _info.get("reasoning", "")
            if _reason:
                lines.append(f"  Verdict: {_reason}")
            _bull = _info.get("bull_thesis", "")
            if _bull:
                lines.append(f"  Bull: {_bull[:200]}")
            _bear = _info.get("bear_thesis", "")
            if _bear:
                lines.append(f"  Bear: {_bear[:200]}")
            _risk = _info.get("risk_assessment", "")
            if _risk:
                lines.append(f"  Risk: {_risk[:200]}")
            lines.append("")

    projection_audit = run_meta.get("optimizer_projection_audit", {}) if isinstance(run_meta, dict) else {}
    if isinstance(projection_audit, dict) and projection_audit:
        lines.append("OPTIMIZER PROJECTION AUDIT")
        lines.append("-" * 78)
        pre = projection_audit.get("pre_rebalance_vs_final", {})
        opt = projection_audit.get("optimizer_vs_final", {})
        if isinstance(pre, dict):
            lines.append(
                "PreRebalance->Final: "
                f"l1={_fmt_num(pre.get('l1_distance'))} "
                f"l2={_fmt_num(pre.get('l2_distance'))} "
                f"changed={pre.get('changed_count', 'N/A')} "
                f"dropped={pre.get('dropped_count', 'N/A')}"
            )
        if isinstance(opt, dict) and opt:
            lines.append(
                "Optimizer->Final: "
                f"l1={_fmt_num(opt.get('l1_distance'))} "
                f"l2={_fmt_num(opt.get('l2_distance'))} "
                f"changed={opt.get('changed_count', 'N/A')} "
                f"dropped={opt.get('dropped_count', 'N/A')}"
            )
        notional_drops = projection_audit.get("notional_drops_with_reasons", [])
        if isinstance(notional_drops, list) and notional_drops:
            lines.append("Notional gate drops (ticker/reason):")
            for evt in notional_drops[:15]:
                if not isinstance(evt, dict):
                    continue
                ticker = str(evt.get("ticker", ""))
                reason = str(evt.get("reason_code", "unknown"))
                lines.append(f"{ticker}: {reason}")
            if len(notional_drops) > 15:
                lines.append(f"... +{len(notional_drops) - 15} more")
        lines.append("")

    def _top_table(df: pd.DataFrame, n: int) -> list[str]:
        cols = [
            "score",
            "last_close_cad",
            "ret_60d",
            "ret_120d",
            "vol_60d_ann",
            "avg_dollar_volume_cad",
            "rsi_14",
        ]
        view = df.head(n)[cols].copy()
        view["last_close_cad"] = view["last_close_cad"].map(_fmt_money)
        view["ret_60d"] = view["ret_60d"].map(_fmt_pct)
        view["ret_120d"] = view["ret_120d"].map(_fmt_pct)
        view["vol_60d_ann"] = view["vol_60d_ann"].map(_fmt_pct)
        view["avg_dollar_volume_cad"] = view["avg_dollar_volume_cad"].map(_fmt_money)
        view["score"] = view["score"].map(_fmt_num)
        view["rsi_14"] = view["rsi_14"].map(_fmt_num)
        return view.to_string().splitlines()

    lines.append("TOP SCREENED (by score)")
    lines.append("-" * 78)
    lines.extend(_top_table(screened, n=min(25, len(screened))))
    lines.append("")

    lines.append("PORTFOLIO HOLDINGS (actual vs target weights)")
    lines.append("-" * 78)
    weights_view = weights.copy()
    fx_rate = _to_float(fx_usdcad_rate)
    if fx_rate is not None and fx_rate > 0 and "last_close_cad" in weights_view.columns:
        weights_view["last_close_usd"] = pd.to_numeric(weights_view["last_close_cad"], errors="coerce") / float(fx_rate)
    else:
        weights_view["last_close_usd"] = pd.NA
    if "position_value_cad" not in weights_view.columns and "shares" in weights_view.columns and "last_close_cad" in weights_view.columns:
        weights_view["position_value_cad"] = pd.to_numeric(weights_view["last_close_cad"], errors="coerce") * pd.to_numeric(
            weights_view["shares"], errors="coerce"
        )
    else:
        weights_view["position_value_cad"] = weights_view.get("position_value_cad", pd.NA)
    if fx_rate is not None and fx_rate > 0:
        weights_view["position_value_usd"] = pd.to_numeric(weights_view["position_value_cad"], errors="coerce") / float(fx_rate)
    else:
        weights_view["position_value_usd"] = pd.NA
    weights_view["weight"] = weights_view["weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
    if "actual_weight" in weights_view.columns:
        weights_view["actual_weight"] = weights_view["actual_weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
    if "target_weight" in weights_view.columns:
        weights_view["target_weight"] = weights_view["target_weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
    weights_view["last_close_cad"] = weights_view["last_close_cad"].map(_fmt_money)
    weights_view["last_close_usd"] = weights_view["last_close_usd"].map(_fmt_money)
    if "shares" in weights_view.columns:
        def _fmt_shares(x):
            v = _to_float(x)
            if v is None or (v != v):  # NaN
                return "N/A"
            if v >= 1:
                return str(int(round(v, 0))) if v == round(v, 0) else f"{v:.4f}".rstrip("0").rstrip(".")
            return f"{v:.4f}".rstrip("0").rstrip(".")
        weights_view["shares"] = weights_view["shares"].map(_fmt_shares)
    else:
        weights_view["shares"] = pd.NA
    weights_view["position_value_cad"] = weights_view["position_value_cad"].map(_fmt_money)
    weights_view["position_value_usd"] = weights_view["position_value_usd"].map(_fmt_money)
    weights_view["ret_60d"] = weights_view["ret_60d"].map(_fmt_pct)
    weights_view["vol_60d_ann"] = weights_view["vol_60d_ann"].map(_fmt_pct)
    weights_cols = [
        "actual_weight",
        "target_weight",
        "weight",
        "shares",
        "position_value_cad",
        "position_value_usd",
        "score",
        "last_close_cad",
        "last_close_usd",
        "ret_60d",
        "vol_60d_ann",
        "avg_dollar_volume_cad",
    ]
    # Only include columns that exist to avoid KeyError
    weights_cols = [c for c in weights_cols if c in weights_view.columns]
    lines.extend(weights_view[weights_cols].to_string().splitlines())
    lines.append("")

    # Show target portfolio weights when holdings are empty (so the user
    # always sees what the model recommends even with no open positions).
    if weights.empty and target_weights is not None and not target_weights.empty:
        lines.append("TARGET PORTFOLIO WEIGHTS (recommended)")
        lines.append("-" * 78)
        tw_view = target_weights.copy()
        if fx_rate is not None and fx_rate > 0 and "last_close_cad" in tw_view.columns:
            tw_view["last_close_usd"] = pd.to_numeric(tw_view["last_close_cad"], errors="coerce") / float(fx_rate)
        else:
            tw_view["last_close_usd"] = pd.NA
        tw_view["weight"] = tw_view["weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
        tw_view["last_close_cad"] = tw_view["last_close_cad"].map(_fmt_money) if "last_close_cad" in tw_view.columns else pd.NA
        tw_view["last_close_usd"] = tw_view["last_close_usd"].map(_fmt_money) if "last_close_usd" in tw_view.columns else pd.NA
        tw_view["ret_60d"] = tw_view["ret_60d"].map(_fmt_pct) if "ret_60d" in tw_view.columns else pd.NA
        tw_view["vol_60d_ann"] = tw_view["vol_60d_ann"].map(_fmt_pct) if "vol_60d_ann" in tw_view.columns else pd.NA
        target_cols = ["weight", "score", "last_close_cad", "last_close_usd", "ret_60d", "vol_60d_ann"]
        target_cols = [c for c in target_cols if c in tw_view.columns]
        lines.extend(tw_view[target_cols].to_string().splitlines())
        lines.append("")

    # Determine prediction horizon for labeling predicted returns
    _label_horizon = run_meta.get("label_horizon_days") or (run_meta.get("config", {}).get("label_horizon_days") if isinstance(run_meta, dict) else None)

    if trade_actions:
        lines.append("RECOMMENDED ACTIONS (sell at predicted peak)")
        lines.append("-" * 78)
        for a in trade_actions:
            ticker = _action_value(a, "ticker", "")
            action = _action_value(a, "action", "")
            reason = _action_value(a, "reason", "")
            shares = _action_value(a, "shares", "")
            px = _action_value(a, "price_cad", None)
            days = _action_value(a, "days_held", None)
            pred_ret = _action_value(a, "pred_return", None)
            sell_date = _action_value(a, "expected_sell_date", "")
            
            px_f = _to_float(px)
            px_cad_str = _fmt_money(px_f) if px_f is not None else "N/A"
            px_usd_str = _fmt_money(px_f / fx_rate) if px_f is not None and fx_rate and fx_rate > 0 else "N/A"
            
            # For SELL actions, show realized gain/loss; for BUY/HOLD show predicted return and target sell price
            if action in ("SELL", "SELL_PARTIAL"):
                entry_px = _action_value(a, "entry_price", None)
                realized_gain = _action_value(a, "realized_gain_pct", None)
                entry_px_f = _to_float(entry_px)
                entry_px_str = _fmt_money(entry_px_f) if entry_px_f is not None else "N/A"
                entry_px_usd_str = _fmt_money(entry_px_f / fx_rate) if entry_px_f is not None and fx_rate and fx_rate > 0 else "N/A"
                realized_str = _fmt_pct(realized_gain) if realized_gain is not None else "N/A"
                days_label = days if days is not None else "N/A"
                lines.append(f"{action:>4} {ticker:<12} shares={shares} sell@={px_cad_str}/{px_usd_str} entry@={entry_px_str}/{entry_px_usd_str} gain={realized_str} days_held={days_label} reason={reason}")
            else:
                pred_ret_str = _fmt_pct(pred_ret) if pred_ret is not None else "N/A"
                _txt_pred_label = f"pred_ret({_label_horizon}d)" if _label_horizon else "pred_ret"
                # Calculate target sell price based on predicted return
                sell_px_cad = float(px_f) * (1 + float(pred_ret)) if px_f is not None and pred_ret is not None else None
                sell_px_cad_str = _fmt_money(sell_px_cad) if sell_px_cad is not None else "N/A"
                sell_px_usd_str = _fmt_money(sell_px_cad / fx_rate) if sell_px_cad and fx_rate and fx_rate > 0 else "N/A"
                lines.append(f"{action:>4} {ticker:<12} shares={shares} price={px_cad_str}/{px_usd_str} {_txt_pred_label}={pred_ret_str} sell@={sell_px_cad_str}/{sell_px_usd_str} sell_date={sell_date or 'N/A'} reason={reason}")
        lines.append("")

    # Portfolio P&L history (stateful; computed from portfolio state positions)
    if portfolio_pnl_history:
        lines.append("PORTFOLIO RETURNS")
        lines.append("-" * 78)
        latest = portfolio_pnl_history[-1]
        prev = portfolio_pnl_history[-2] if len(portfolio_pnl_history) >= 2 else None
        first = portfolio_pnl_history[0]
        
        latest_asof = latest.get("asof_utc")
        prev_asof = prev.get("asof_utc") if prev else None
        equity = _to_float(latest.get("equity_cad"))
        prev_equity = _to_float(prev.get("equity_cad")) if prev else None
        first_equity = _to_float(first.get("equity_cad"))
        cash = _to_float(latest.get("cash_cad"))
        open_market_value = _to_float(latest.get("open_market_value_cad"))
        realized_pl = _to_float(latest.get("realized_pl_cad"))
        unrealized_pl = _to_float(latest.get("unrealized_pl_cad"))
        net_pl = _to_float(latest.get("net_pl_cad"))
        
        # Calculate returns
        all_time_return = None
        day_to_day_return = None

        if equity is not None and first_equity is not None and first_equity > 0:
            all_time_return = (equity - first_equity) / first_equity

        if equity is not None and prev_equity is not None and prev_equity > 0:
            day_to_day_return = (equity - prev_equity) / prev_equity

        # Compute time period labels for returns
        all_time_label = "All-Time Return"
        day_to_day_label = "Day-to-Day Return"
        first_asof = first.get("asof_utc")
        if first_asof and latest_asof:
            try:
                from datetime import datetime as _dt
                _fmt = "%Y-%m-%dT%H:%M:%S" if "T" in str(first_asof) else "%Y-%m-%d"
                _first_dt = _dt.fromisoformat(str(first_asof).replace("Z", "+00:00")) if hasattr(_dt, "fromisoformat") else _dt.strptime(str(first_asof)[:10], "%Y-%m-%d")
                _latest_dt = _dt.fromisoformat(str(latest_asof).replace("Z", "+00:00")) if hasattr(_dt, "fromisoformat") else _dt.strptime(str(latest_asof)[:10], "%Y-%m-%d")
                _n_days = (_latest_dt - _first_dt).days
                all_time_label = f"All-Time Return ({_n_days}d, since {str(first_asof)[:10]})"
            except Exception:
                pass
        if prev_asof and latest_asof:
            try:
                day_to_day_label = f"Day-to-Day Return ({str(prev_asof)[:10]} to {str(latest_asof)[:10]})"
            except Exception:
                pass

        if latest_asof:
            lines.append(f"Snapshot: {latest_asof}")
        if prev_asof:
            lines.append(f"Previous Snapshot: {prev_asof}")
        lines.append(f"Current Equity: {_fmt_money(equity) if equity is not None else 'N/A'}")
        if fx_rate is not None and fx_rate > 0 and equity is not None:
            lines.append(f"Current Equity USD: {_fmt_money(equity / fx_rate)}")
        lines.append(f"Cash: {_fmt_money(cash) if cash is not None else 'N/A'}")
        lines.append(f"Invested Market Value: {_fmt_money(open_market_value) if open_market_value is not None else 'N/A'}")
        lines.append(f"Realized P&L: {_fmt_money(realized_pl) if realized_pl is not None else 'N/A'}")
        lines.append(f"Unrealized P&L: {_fmt_money(unrealized_pl) if unrealized_pl is not None else 'N/A'}")
        lines.append(f"Net P&L: {_fmt_money(net_pl) if net_pl is not None else 'N/A'}")
        lines.append("")
        lines.append(f"{all_time_label}: {_fmt_pct(all_time_return) if all_time_return is not None else 'N/A'}")
        lines.append(f"{day_to_day_label}: {_fmt_pct(day_to_day_return) if day_to_day_return is not None else 'N/A'}")
        lines.append("")

    model_block = ""
    if model_metrics:
        reg_summary = model_metrics.get("regressor", {}).get("holdout") or {}
        model_block = f"""
  <div style="background:#ecfeff;border-radius:8px;padding:12px 14px;margin: 0 0 18px 0;">
    <div style="font-weight:600;margin-bottom:6px;">Model Validation (Holdout IC)</div>
    <table style="border-collapse: collapse; width: 100%; font-size: 13px;">
      <thead>
        <tr>
          <th style="text-align:left;padding:4px 6px;border-bottom:1px solid #0ea5e9;">Model</th>
          <th style="text-align:left;padding:4px 6px;border-bottom:1px solid #0ea5e9;">Mean IC</th>
          <th style="text-align:left;padding:4px 6px;border-bottom:1px solid #0ea5e9;">Std IC</th>
          <th style="text-align:left;padding:4px 6px;border-bottom:1px solid #0ea5e9;">IC IR</th>
          <th style="text-align:left;padding:4px 6px;border-bottom:1px solid #0ea5e9;">N Days</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding:4px 6px;">Regressor</td>
          <td style="padding:4px 6px;">{_fmt_num(reg_summary.get("mean_ic"))}</td>
          <td style="padding:4px 6px;">{_fmt_num(reg_summary.get("std_ic"))}</td>
          <td style="padding:4px 6px;">{_fmt_num(reg_summary.get("ic_ir"))}</td>
          <td style="padding:4px 6px;">{reg_summary.get("n_days", "N/A")}</td>
        </tr>
      </tbody>
    </table>
  </div>
"""

    lines.append("FILES")
    lines.append("-" * 78)
    lines.append("reports/daily_email.html")
    lines.append("reports/daily_report.txt")
    lines.append("reports/portfolio_weights.csv")
    if trade_actions:
        lines.append("reports/trade_actions.json")
    lines.append("")
    (reports_dir / "daily_report.txt").write_text("\n".join(lines), encoding="utf-8")

    # HTML email (simple and robust: no external templating dependency)
    _base_cols = [
        "ticker", "actual_weight", "target_weight", "weight", "shares",
        "position_value_cad", "score", "last_close_cad", "ret_60d", "vol_60d_ann",
    ]
    _reset = weights.reset_index()
    if "ticker" not in _reset.columns and len(_reset.columns) > 0:
        _reset = _reset.rename(columns={_reset.columns[0]: "ticker"})
    _base_cols = [c for c in _base_cols if c in _reset.columns]
    weights_table = _reset[_base_cols].copy()
    # Ensure columns exist (may be missing for holdings built from raw features)
    for _c in ["score", "ret_60d", "vol_60d_ann", "last_close_cad"]:
        if _c not in weights_table.columns:
            weights_table[_c] = pd.NA
    fx_rate = _to_float(fx_usdcad_rate)
    if fx_rate is not None and fx_rate > 0 and "last_close_cad" in weights_table.columns:
        weights_table["last_close_usd"] = pd.to_numeric(weights_table["last_close_cad"], errors="coerce") / float(fx_rate)
    else:
        weights_table["last_close_usd"] = pd.NA
    if "shares" in weights.columns:
        shares = weights["shares"].copy()
        shares.index = weights.index.astype(str)
        weights_table["shares"] = weights_table["ticker"].astype(str).map(shares)
    else:
        weights_table["shares"] = pd.NA
    if "position_value_cad" in weights.columns:
        pos_val = weights["position_value_cad"].copy()
        pos_val.index = weights.index.astype(str)
        weights_table["position_value_cad"] = weights_table["ticker"].astype(str).map(pos_val)
    else:
        weights_table["position_value_cad"] = pd.to_numeric(weights_table["last_close_cad"], errors="coerce") * pd.to_numeric(
            weights_table["shares"], errors="coerce"
        )
    if fx_rate is not None and fx_rate > 0:
        weights_table["position_value_usd"] = pd.to_numeric(weights_table["position_value_cad"], errors="coerce") / float(fx_rate)
    else:
        weights_table["position_value_usd"] = pd.NA
    if "actual_weight" in weights_table.columns:
        weights_table["actual_weight"] = weights_table["actual_weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
    if "target_weight" in weights_table.columns:
        weights_table["target_weight"] = weights_table["target_weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
    weights_table["weight"] = weights_table["weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
    weights_table["last_close_cad"] = weights_table["last_close_cad"].map(_fmt_money)
    weights_table["last_close_usd"] = weights_table["last_close_usd"].map(_fmt_money)
    def _fmt_shares(x):
        v = _to_float(x)
        if v is None or (v != v):  # NaN
            return "N/A"
        if v >= 1:
            return str(int(round(v, 0))) if v == round(v, 0) else f"{v:.4f}".rstrip("0").rstrip(".")
        return f"{v:.4f}".rstrip("0").rstrip(".")
    weights_table["shares"] = weights_table["shares"].map(_fmt_shares)
    weights_table["position_value_cad"] = weights_table["position_value_cad"].map(_fmt_money)
    weights_table["position_value_usd"] = weights_table["position_value_usd"].map(_fmt_money)
    weights_table["ret_60d"] = weights_table["ret_60d"].map(_fmt_pct)
    weights_table["vol_60d_ann"] = weights_table["vol_60d_ann"].map(_fmt_pct)
    weights_table["score"] = weights_table["score"].map(_fmt_num)
    _html_cols = [
        "ticker", "actual_weight", "target_weight", "weight", "shares",
        "position_value_cad", "position_value_usd", "score",
        "last_close_cad", "last_close_usd", "ret_60d", "vol_60d_ann",
    ]
    _html_cols = [c for c in _html_cols if c in weights_table.columns]
    weights_table = weights_table[_html_cols].copy()
    col_labels = {
        "ticker": "Ticker",
        "actual_weight": "Actual Weight",
        "target_weight": "Target Weight",
        "weight": "Weight",
        "shares": "Shares",
        "position_value_cad": "Value (CAD)",
        "position_value_usd": "Value (USD)",
        "score": "Score",
        "last_close_cad": "Price (CAD)",
        "last_close_usd": "Price (USD)",
        "ret_60d": "Ret 60d",
        "vol_60d_ann": "Vol 60d (ann)",
    }
    headers_html = "".join(
        f"<th style='text-align:left;padding:6px 8px;border-bottom:2px solid #111827;'>{_html_escape(col_labels.get(c, c))}</th>"
        for c in weights_table.columns
    )

    rows_html = "\n".join(
        "<tr>"
        + "".join(f"<td style='padding:6px 8px;border-bottom:1px solid #e5e7eb;'>{_html_escape(str(v))}</td>" for v in row)
        + "</tr>"
        for row in weights_table.itertuples(index=False, name=None)
    )

    # Build actions block outside the f-string to avoid complex nested expressions.
    _pred_ret_label = f"Pred Ret ({_label_horizon}d)" if _label_horizon else "Pred Ret"

    fx_rate = _to_float(fx_usdcad_rate)
    if trade_actions:
        # Separate actions by type for card rendering
        sell_actions = [a for a in trade_actions if (getattr(a, "action", None) or (a.get("action") if isinstance(a, dict) else "")) in ("SELL", "SELL_PARTIAL")]
        buy_actions = [a for a in trade_actions if (getattr(a, "action", None) or (a.get("action") if isinstance(a, dict) else "")) == "BUY"]
        hold_actions = [a for a in trade_actions if (getattr(a, "action", None) or (a.get("action") if isinstance(a, dict) else "")) not in ("SELL", "SELL_PARTIAL", "BUY")]

        action_cards: list[str] = []

        # --- SELL action cards (red left-border) ---
        for a in sell_actions:
            ticker = _action_value(a, "ticker", "")
            action = _action_value(a, "action", "")
            reason = _action_value(a, "reason", "")
            shares = _action_value(a, "shares", "")
            px = _action_value(a, "price_cad", None)
            days = _action_value(a, "days_held", None)
            entry_px = _action_value(a, "entry_price", None)
            realized_gain = _action_value(a, "realized_gain_pct", None)

            px_f = _to_float(px)
            entry_px_f = _to_float(entry_px)
            px_cad_str = _fmt_money(px_f) if px_f is not None else "N/A"
            px_usd_str = _fmt_money(px_f / fx_rate) if px_f is not None and fx_rate and fx_rate > 0 else "N/A"
            entry_px_str = _fmt_money(entry_px_f) if entry_px_f is not None else "N/A"
            entry_px_usd_str = _fmt_money(entry_px_f / fx_rate) if entry_px_f is not None and fx_rate and fx_rate > 0 else "N/A"
            realized_str = _fmt_pct(realized_gain) if realized_gain is not None else "N/A"
            gain_color = "#059669" if realized_gain and realized_gain > 0 else "#dc2626" if realized_gain and realized_gain < 0 else "#6b7280"
            gain_arrow = "&#9650;" if realized_gain and realized_gain > 0 else "&#9660;" if realized_gain and realized_gain < 0 else ""
            days_label = str(days) if days is not None else "N/A"

            action_cards.append(
                f'<div style="background:#ffffff;border-radius:12px;border-left:5px solid #dc2626;'
                f'padding:14px 16px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.08);">'
                # Row 1: badge + ticker + gain
                f'<table style="width:100%;border-collapse:collapse;"><tr>'
                f'<td style="vertical-align:middle;width:1%;white-space:nowrap;padding:0 10px 0 0;">'
                f'<span style="background:#dc2626;color:#ffffff;font-size:11px;font-weight:700;'
                f'padding:3px 10px;border-radius:6px;letter-spacing:0.5px;">{_html_escape(action)}</span></td>'
                f'<td style="vertical-align:middle;padding:0;">'
                f'<span style="font-size:20px;font-weight:800;color:#111827;letter-spacing:-0.5px;">{_html_escape(str(ticker))}</span></td>'
                f'<td style="vertical-align:middle;text-align:right;padding:0;">'
                f'<span style="font-size:18px;font-weight:700;color:{gain_color};">{gain_arrow} {realized_str}</span></td>'
                f'</tr></table>'
                # Row 2: details
                f'<table style="width:100%;border-collapse:collapse;margin-top:10px;font-size:12px;color:#6b7280;"><tr>'
                f'<td style="padding:2px 0;"><span style="color:#9ca3af;">Entry</span><br/>'
                f'<span style="color:#111827;font-weight:600;">${entry_px_str}</span>'
                f'<span style="color:#9ca3af;font-size:11px;"> / ${entry_px_usd_str}</span></td>'
                f'<td style="padding:2px 0;"><span style="color:#9ca3af;">Exit</span><br/>'
                f'<span style="color:#111827;font-weight:600;">${px_cad_str}</span>'
                f'<span style="color:#9ca3af;font-size:11px;"> / ${px_usd_str}</span></td>'
                f'<td style="padding:2px 0;"><span style="color:#9ca3af;">Shares</span><br/>'
                f'<span style="color:#111827;font-weight:600;">{shares}</span></td>'
                f'<td style="padding:2px 0;"><span style="color:#9ca3af;">Held</span><br/>'
                f'<span style="color:#111827;font-weight:600;">{days_label}d</span></td>'
                f'</tr></table>'
                # Row 3: reason
                f'<div style="margin-top:10px;padding:8px 10px;background:#fef2f2;border-radius:8px;'
                f'font-size:12px;color:#991b1b;line-height:1.4;">{_html_escape(str(reason))}</div>'
                f'</div>'
            )

        # --- BUY action cards (green left-border) ---
        for a in buy_actions:
            ticker = _action_value(a, "ticker", "")
            action = _action_value(a, "action", "")
            reason = _action_value(a, "reason", "")
            shares = _action_value(a, "shares", "")
            px = _action_value(a, "price_cad", None)
            pred_ret = _action_value(a, "pred_return", None)
            sell_date = _action_value(a, "expected_sell_date", "")

            pred_ret_str = _fmt_pct(pred_ret) if pred_ret is not None else "N/A"
            px_f = _to_float(px)
            px_cad_str = _fmt_money(px_f) if px_f is not None else "N/A"
            px_usd_str = _fmt_money(px_f / fx_rate) if px_f is not None and fx_rate and fx_rate > 0 else "N/A"
            sell_px_cad = float(px_f) * (1 + float(pred_ret)) if px_f is not None and pred_ret is not None else None
            sell_px_cad_str = _fmt_money(sell_px_cad) if sell_px_cad is not None else "N/A"
            sell_px_usd_str = _fmt_money(sell_px_cad / fx_rate) if sell_px_cad and fx_rate and fx_rate > 0 else "N/A"
            ret_color = "#059669" if pred_ret and pred_ret > 0 else "#dc2626" if pred_ret and pred_ret < 0 else "#6b7280"

            action_cards.append(
                f'<div style="background:#ffffff;border-radius:12px;border-left:5px solid #059669;'
                f'padding:14px 16px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.08);">'
                # Row 1: badge + ticker + predicted return
                f'<table style="width:100%;border-collapse:collapse;"><tr>'
                f'<td style="vertical-align:middle;width:1%;white-space:nowrap;padding:0 10px 0 0;">'
                f'<span style="background:#059669;color:#ffffff;font-size:11px;font-weight:700;'
                f'padding:3px 10px;border-radius:6px;letter-spacing:0.5px;">{_html_escape(action)}</span></td>'
                f'<td style="vertical-align:middle;padding:0;">'
                f'<span style="font-size:20px;font-weight:800;color:#111827;letter-spacing:-0.5px;">{_html_escape(str(ticker))}</span></td>'
                f'<td style="vertical-align:middle;text-align:right;padding:0;">'
                f'<span style="font-size:13px;font-weight:600;color:{ret_color};">{_html_escape(_pred_ret_label)}: {pred_ret_str}</span></td>'
                f'</tr></table>'
                # Row 2: details
                f'<table style="width:100%;border-collapse:collapse;margin-top:10px;font-size:12px;color:#6b7280;"><tr>'
                f'<td style="padding:2px 0;"><span style="color:#9ca3af;">Buy @</span><br/>'
                f'<span style="color:#111827;font-weight:600;">${px_cad_str}</span>'
                f'<span style="color:#9ca3af;font-size:11px;"> / ${px_usd_str}</span></td>'
                f'<td style="padding:2px 0;"><span style="color:#9ca3af;">Target Sell @</span><br/>'
                f'<span style="color:#059669;font-weight:700;">${sell_px_cad_str}</span>'
                f'<span style="color:#9ca3af;font-size:11px;"> / ${sell_px_usd_str}</span></td>'
                f'<td style="padding:2px 0;"><span style="color:#9ca3af;">Shares</span><br/>'
                f'<span style="color:#111827;font-weight:600;">{shares}</span></td>'
                f'<td style="padding:2px 0;"><span style="color:#9ca3af;">Sell Date</span><br/>'
                f'<span style="color:#111827;font-weight:600;">{_html_escape(str(sell_date)) if sell_date else "N/A"}</span></td>'
                f'</tr></table>'
                # Row 3: reason
                f'<div style="margin-top:10px;padding:8px 10px;background:#ecfdf5;border-radius:8px;'
                f'font-size:12px;color:#065f46;line-height:1.4;">{_html_escape(str(reason))}</div>'
                f'</div>'
            )

        # --- HOLD action cards (blue left-border, compact) ---
        for a in hold_actions:
            ticker = _action_value(a, "ticker", "")
            action = _action_value(a, "action", "")
            reason = _action_value(a, "reason", "")
            shares = _action_value(a, "shares", "")
            px = _action_value(a, "price_cad", None)
            pred_ret = _action_value(a, "pred_return", None)
            sell_date = _action_value(a, "expected_sell_date", "")

            pred_ret_str = _fmt_pct(pred_ret) if pred_ret is not None else "N/A"
            px_f = _to_float(px)
            px_cad_str = _fmt_money(px_f) if px_f is not None else "N/A"
            px_usd_str = _fmt_money(px_f / fx_rate) if px_f is not None and fx_rate and fx_rate > 0 else "N/A"
            sell_px_cad = float(px_f) * (1 + float(pred_ret)) if px_f is not None and pred_ret is not None else None
            sell_px_cad_str = _fmt_money(sell_px_cad) if sell_px_cad is not None else "N/A"
            sell_px_usd_str = _fmt_money(sell_px_cad / fx_rate) if sell_px_cad and fx_rate and fx_rate > 0 else "N/A"

            action_cards.append(
                f'<div style="background:#ffffff;border-radius:10px;border-left:4px solid #2563eb;'
                f'padding:10px 14px;margin-bottom:8px;box-shadow:0 1px 2px rgba(0,0,0,0.05);">'
                f'<table style="width:100%;border-collapse:collapse;font-size:12px;"><tr>'
                f'<td style="vertical-align:middle;width:1%;white-space:nowrap;padding:0 8px 0 0;">'
                f'<span style="background:#2563eb;color:#ffffff;font-size:10px;font-weight:700;'
                f'padding:2px 8px;border-radius:5px;">{_html_escape(action)}</span></td>'
                f'<td style="vertical-align:middle;padding:0;">'
                f'<span style="font-size:15px;font-weight:700;color:#111827;">{_html_escape(str(ticker))}</span></td>'
                f'<td style="vertical-align:middle;padding:0;color:#6b7280;">'
                f'{shares} shares @ ${px_cad_str}'
                f'<span style="color:#9ca3af;font-size:11px;"> / ${px_usd_str}</span></td>'
                f'<td style="vertical-align:middle;padding:0;color:#6b7280;white-space:nowrap;">'
                f'{_html_escape(_pred_ret_label)}: <span style="font-weight:600;">{pred_ret_str}</span></td>'
                f'<td style="vertical-align:middle;padding:0;color:#6b7280;white-space:nowrap;">'
                f'Target: <span style="font-weight:600;">${sell_px_cad_str}</span>'
                f'<span style="color:#9ca3af;font-size:11px;"> / ${sell_px_usd_str}</span></td>'
                f'<td style="vertical-align:middle;padding:0;color:#6b7280;white-space:nowrap;">'
                f'Sell: {_html_escape(str(sell_date)) if sell_date else "N/A"}</td>'
                f'</tr></table>'
                f'<div style="margin-top:6px;font-size:11px;color:#6b7280;line-height:1.3;">{_html_escape(str(reason))}</div>'
                f'</div>'
            )

        if action_cards:
            actions_html = "\n".join(action_cards)
        else:
            actions_html = (
                '<div style="background:#ffffff;border-radius:12px;padding:20px;text-align:center;'
                'color:#9ca3af;font-size:13px;box-shadow:0 1px 3px rgba(0,0,0,0.08);">'
                'No actions &mdash; portfolio already aligned.</div>'
            )
    else:
        actions_html = (
            '<div style="background:#ffffff;border-radius:12px;padding:20px;text-align:center;'
            'color:#9ca3af;font-size:13px;box-shadow:0 1px 3px rgba(0,0,0,0.08);">'
            'No actions &mdash; portfolio already aligned.</div>'
        )

    # LLM Agent Analysis block
    llm_block = ""
    llm_agent_data = run_meta.get("llm_agent") if isinstance(run_meta, dict) else None
    if isinstance(llm_agent_data, dict) and not llm_agent_data.get("decisions"):
        _llm_status = llm_agent_data.get("status", "unknown")
        _llm_reason = llm_agent_data.get("reason", "")
        if _llm_status != "disabled":
            _status_label = {"skipped": "Skipped", "no_results": "No Results", "error": "Error"}.get(_llm_status, _llm_status)
            llm_block = f"""
  <div style="background:#fef3c7;border-radius:8px;padding:10px 14px;margin:0 0 18px 0;font-size:13px;">
    <strong>LLM Agent:</strong> {_html_escape(_status_label)}{(' — ' + _html_escape(_llm_reason)) if _llm_reason else ''}
  </div>
"""
    if isinstance(llm_agent_data, dict) and llm_agent_data.get("decisions"):
        ticker_cards_html = ""
        for ticker, info in llm_agent_data["decisions"].items():
            if not isinstance(info, dict):
                continue
            rating = info.get("rating", "N/A")
            score = info.get("score", 0)
            reasoning = info.get("reasoning", "")
            bull = info.get("bull_thesis", "")
            bear = info.get("bear_thesis", "")
            risk = info.get("risk_assessment", "")
            risk_debate = info.get("risk_debate")
            analyst_reports = info.get("analyst_reports")
            debate_history = info.get("debate_history", [])
            debate_rounds = info.get("debate_rounds", 1)
            rating_colors = {"BUY": "#059669", "OVERWEIGHT": "#10b981", "HOLD": "#6b7280", "UNDERWEIGHT": "#f59e0b", "SELL": "#dc2626"}
            rating_bg = {"BUY": "#ecfdf5", "OVERWEIGHT": "#ecfdf5", "HOLD": "#f3f4f6", "UNDERWEIGHT": "#fffbeb", "SELL": "#fef2f2"}
            rc = rating_colors.get(rating, "#6b7280")
            rb = rating_bg.get(rating, "#f3f4f6")

            # ── Analyst Team cards (Phase 3) ──
            analyst_section = ""
            if isinstance(analyst_reports, dict) and any(analyst_reports.values()):
                analyst_cards = ""
                analyst_defs = [
                    ("technical", "Technical", "#f59e0b", "&#128200;"),
                    ("fundamental", "Fundamental", "#f59e0b", "&#128176;"),
                    ("sentiment", "Sentiment", "#f59e0b", "&#128240;"),
                ]
                for key, label, accent, icon in analyst_defs:
                    report = analyst_reports.get(key, "")
                    if not report:
                        continue
                    analyst_cards += f"""
              <div style="flex:1;min-width:200px;background:#1e1e2e;border-radius:10px;padding:12px;border-top:3px solid {accent};">
                <div style="font-size:11px;color:{accent};font-weight:bold;margin-bottom:6px;">{icon} {label}</div>
                <div style="color:#e0e0e0;font-size:11px;line-height:1.5;">{_html_escape(report)}</div>
              </div>"""
                if analyst_cards:
                    analyst_section = f"""
            <details style="margin-bottom:10px;">
              <summary style="cursor:pointer;font-weight:bold;font-size:12px;color:#f59e0b;margin-bottom:6px;">&#128269; Analyst Team Reports</summary>
              <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;">{analyst_cards}
              </div>
            </details>"""

            # ── Bull vs Bear Debate cards (Phase 1) ──
            debate_section = ""
            if bull or bear:
                debate_rounds_html = ""
                if isinstance(debate_history, list) and len(debate_history) > 2:
                    for i, (side, text) in enumerate(debate_history):
                        round_num = (i // 2) + 1
                        side_color = "#059669" if side == "BULL" else "#dc2626"
                        side_icon = "&#128200;" if side == "BULL" else "&#128201;"
                        debate_rounds_html += f"""
                  <div style="padding:6px 10px;font-size:11px;border-left:3px solid {side_color};margin-bottom:4px;background:{'#ecfdf5' if side == 'BULL' else '#fef2f2'};">
                    <strong style="color:{side_color};">{side_icon} R{round_num} {side}:</strong> {_html_escape(text)}
                  </div>"""
                    debate_rounds_html = f"""
                <details style="margin-top:6px;">
                  <summary style="cursor:pointer;font-size:10px;color:#6b7280;">Full debate ({len(debate_history)} exchanges)</summary>
                  <div style="margin-top:4px;">{debate_rounds_html}</div>
                </details>"""

                debate_section = f"""
            <div style="display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap;">
              <div style="flex:1;min-width:220px;background:#ecfdf5;border-radius:10px;padding:12px;border-left:4px solid #059669;">
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
                  <span style="background:#059669;color:white;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold;">&#128200; Bullish</span>
                </div>
                <div style="font-size:11px;color:#374151;font-weight:bold;margin-bottom:4px;">Investment Opportunity</div>
                <div style="font-size:11px;color:#374151;line-height:1.5;">{_html_escape(bull)}</div>
              </div>
              <div style="flex:0 0 40px;display:flex;align-items:center;justify-content:center;font-size:11px;color:#9ca3af;flex-direction:column;">
                <div>&#8594;</div><div style="font-size:10px;">Debate</div><div>&#8592;</div>
              </div>
              <div style="flex:1;min-width:220px;background:#fef2f2;border-radius:10px;padding:12px;border-left:4px solid #dc2626;">
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
                  <span style="background:#dc2626;color:white;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold;">&#128201; Bearish</span>
                </div>
                <div style="font-size:11px;color:#374151;font-weight:bold;margin-bottom:4px;">Investment Risks</div>
                <div style="font-size:11px;color:#374151;line-height:1.5;">{_html_escape(bear)}</div>
              </div>
            </div>{debate_rounds_html}"""

            # ── Risk Management Team (Phase 2) ──
            risk_section = ""
            if isinstance(risk_debate, dict) and any(risk_debate.values()):
                risk_defs = [
                    ("aggressive", "Risky", "#f97316", "&#128293;"),
                    ("neutral", "Neutral", "#3b82f6", "&#9878;"),
                    ("conservative", "Safe", "#22c55e", "&#128737;"),
                ]
                risk_cards = ""
                for key, label, color, icon in risk_defs:
                    view = risk_debate.get(key, "")
                    if not view:
                        continue
                    risk_cards += f"""
                <div style="background:#1e1e2e;border-radius:8px;padding:10px 12px;margin-bottom:6px;border-left:3px solid {color};">
                  <span style="color:{color};font-weight:bold;font-size:11px;">{icon} {label}:</span>
                  <span style="color:#e0e0e0;font-size:11px;"> {_html_escape(view)}</span>
                </div>"""
                risk_section = f"""
            <details style="margin-bottom:10px;">
              <summary style="cursor:pointer;font-weight:bold;font-size:12px;color:#3b82f6;margin-bottom:6px;">&#128737; Risk Management Debate</summary>
              <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:6px;">
                <div style="flex:1;min-width:300px;">{risk_cards}</div>
                <div style="flex:1;min-width:200px;background:#eff6ff;border-radius:10px;padding:12px;border-left:4px solid #3b82f6;">
                  <div style="font-size:11px;font-weight:bold;color:#1e40af;margin-bottom:4px;">&#128100; Risk Synthesis</div>
                  <div style="font-size:11px;color:#374151;line-height:1.5;">{_html_escape(risk)}</div>
                </div>
              </div>
            </details>"""
            elif risk:
                risk_section = f"""
            <details style="margin-bottom:6px;">
              <summary style="cursor:pointer;font-size:12px;font-weight:bold;color:#3b82f6;">&#128737; Risk Assessment</summary>
              <div style="padding:8px 12px;font-size:11px;color:#374151;background:#eff6ff;border-radius:8px;margin-top:4px;line-height:1.5;">{_html_escape(risk)}</div>
            </details>"""

            # ── Assemble ticker card (collapsible, shows ticker+rating+verdict) ──
            _verdict_preview = _html_escape(reasoning[:120]) + ("..." if len(reasoning) > 120 else "") if reasoning else ""
            ticker_cards_html += f"""
          <div style="background:white;border-radius:12px;border:1px solid #e5e7eb;padding:0;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,0.06);overflow:hidden;">
            <details>
              <summary style="cursor:pointer;padding:14px 16px;list-style:none;margin:0;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                  <div>
                    <span style="font-size:16px;font-weight:bold;color:#111827;">{_html_escape(str(ticker))}</span>
                    <span style="background:{rb};color:{rc};font-weight:bold;padding:3px 10px;border-radius:6px;font-size:12px;margin-left:8px;">{_html_escape(rating)} ({score:+.1f})</span>
                    <span style="font-size:11px;color:#9ca3af;margin-left:8px;">{'%d-round debate' % debate_rounds if debate_rounds > 1 else ''}</span>
                  </div>
                  <div style="font-size:11px;color:#6b7280;">&#9654; expand</div>
                </div>
                <div style="font-size:12px;color:#374151;margin-top:6px;border-left:4px solid {rc};padding-left:10px;">{_verdict_preview}</div>
              </summary>
              <div style="padding:0 16px 16px 16px;">
                <div style="background:#f8fafc;border-radius:8px;padding:10px 12px;margin-bottom:12px;font-size:12px;color:#374151;border-left:4px solid {rc};">
                  <strong>Full Verdict:</strong> {_html_escape(reasoning)}
                </div>
                {analyst_section}
                {debate_section}
                {risk_section}
              </div>
            </details>
          </div>"""

        if ticker_cards_html:
            n_analyzed = llm_agent_data.get("n_analyzed", 0)
            # Portfolio reasoning (Phase 4)
            portfolio_reasoning_html = ""
            portfolio_reasoning = llm_agent_data.get("portfolio_reasoning")
            if isinstance(portfolio_reasoning, dict):
                pr_items = ""
                for key, label, icon in [
                    ("concentration_risk", "Concentration Risk", "&#128202;"),
                    ("correlation_flag", "Correlation Flag", "&#128279;"),
                    ("regime_check", "Regime Check", "&#127777;"),
                    ("adjustments", "Weight Adjustments", "&#9878;"),
                    ("overall", "Overall Assessment", "&#128161;"),
                ]:
                    val = portfolio_reasoning.get(key, "")
                    if val:
                        pr_items += f'<div style="padding:4px 0;font-size:11px;"><strong>{icon} {label}:</strong> {_html_escape(val)}</div>'
                if pr_items:
                    portfolio_reasoning_html = f"""
    <div style="background:#faf5ff;border-radius:10px;padding:12px 14px;margin-bottom:14px;border-left:4px solid #8b5cf6;">
      <div style="font-weight:bold;font-size:13px;color:#6d28d9;margin-bottom:8px;">&#128300; Portfolio-Level Reasoning</div>
      {pr_items}
    </div>"""

            llm_block = f"""
  <h3 style="margin:0 0 12px 0;">&#129302; Multi-Agent Trading Analysis ({n_analyzed} tickers)</h3>
  <div style="background:#f8fafc;border-radius:12px;padding:14px;margin:0 0 18px 0;">
    <div style="display:flex;gap:14px;margin-bottom:14px;flex-wrap:wrap;">
      <div style="background:#1e1e2e;color:#e0e0e0;border-radius:8px;padding:8px 14px;font-size:11px;flex:1;min-width:120px;text-align:center;">
        <div style="color:#f59e0b;font-weight:bold;margin-bottom:2px;">Analyst Team</div>
        <div>Technical + Fundamental + Sentiment</div>
      </div>
      <div style="color:#9ca3af;display:flex;align-items:center;font-size:16px;">&#8594;</div>
      <div style="background:#1e1e2e;color:#e0e0e0;border-radius:8px;padding:8px 14px;font-size:11px;flex:1;min-width:120px;text-align:center;">
        <div style="font-weight:bold;margin-bottom:2px;"><span style="color:#059669;">Bull</span> vs <span style="color:#dc2626;">Bear</span></div>
        <div>Researcher Debate</div>
      </div>
      <div style="color:#9ca3af;display:flex;align-items:center;font-size:16px;">&#8594;</div>
      <div style="background:#1e1e2e;color:#e0e0e0;border-radius:8px;padding:8px 14px;font-size:11px;flex:1;min-width:120px;text-align:center;">
        <div style="font-weight:bold;margin-bottom:2px;"><span style="color:#f97316;">Risky</span> / <span style="color:#3b82f6;">Neutral</span> / <span style="color:#22c55e;">Safe</span></div>
        <div>Risk Management</div>
      </div>
      <div style="color:#9ca3af;display:flex;align-items:center;font-size:16px;">&#8594;</div>
      <div style="background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:white;border-radius:8px;padding:8px 14px;font-size:11px;flex:1;min-width:100px;text-align:center;">
        <div style="font-weight:bold;margin-bottom:2px;">&#128161; Decision</div>
        <div>Portfolio Manager</div>
      </div>
    </div>
    {portfolio_reasoning_html}
    {ticker_cards_html}
  </div>
"""

    pnl_block = ""
    # Variables used by executive summary cards later
    _pnl_equity = None
    _pnl_day_change = None
    _pnl_all_time_return = None
    if portfolio_pnl_history:
        latest = portfolio_pnl_history[-1]
        prev = portfolio_pnl_history[-2] if len(portfolio_pnl_history) >= 2 else None
        first = portfolio_pnl_history[0]

        latest_asof = latest.get("asof_utc")
        prev_asof = prev.get("asof_utc") if prev else None
        equity = _to_float(latest.get("equity_cad"))
        prev_equity = _to_float(prev.get("equity_cad")) if prev else None
        first_equity = _to_float(first.get("equity_cad"))
        cash = _to_float(latest.get("cash_cad"))
        open_market_value = _to_float(latest.get("open_market_value_cad"))
        realized_pl = _to_float(latest.get("realized_pl_cad"))
        unrealized_pl = _to_float(latest.get("unrealized_pl_cad"))
        net_pl = _to_float(latest.get("net_pl_cad"))

        # Calculate returns
        all_time_return = None
        day_to_day_return = None

        if equity is not None and first_equity is not None and first_equity > 0:
            all_time_return = (equity - first_equity) / first_equity

        if equity is not None and prev_equity is not None and prev_equity > 0:
            day_to_day_return = (equity - prev_equity) / prev_equity

        # Expose for executive summary cards
        _pnl_equity = equity
        _pnl_day_change = day_to_day_return
        _pnl_all_time_return = all_time_return

        # Compute time period labels for returns
        all_time_label_html = "All-Time Return"
        day_to_day_label_html = "Day-to-Day Return"
        first_asof = first.get("asof_utc")
        if first_asof and latest_asof:
            try:
                from datetime import datetime as _dt
                _first_dt = _dt.fromisoformat(str(first_asof).replace("Z", "+00:00")) if hasattr(_dt, "fromisoformat") else _dt.strptime(str(first_asof)[:10], "%Y-%m-%d")
                _latest_dt = _dt.fromisoformat(str(latest_asof).replace("Z", "+00:00")) if hasattr(_dt, "fromisoformat") else _dt.strptime(str(latest_asof)[:10], "%Y-%m-%d")
                _n_days = (_latest_dt - _first_dt).days
                all_time_label_html = f"All-Time Return ({_n_days}d, since {str(first_asof)[:10]})"
            except Exception:
                pass
        if prev_asof and latest_asof:
            try:
                day_to_day_label_html = f"Day-to-Day Return ({str(prev_asof)[:10]} &rarr; {str(latest_asof)[:10]})"
            except Exception:
                pass

        fx_rate = _to_float(fx_usdcad_rate)

        # Build P&L detail rows for the card layout
        def _pnl_row(label: str, val_str: str, color: str = "#111827") -> str:
            return (
                f'<tr><td style="padding:5px 8px;font-size:12px;color:#6b7280;border-bottom:1px solid #f3f4f6;">{label}</td>'
                f'<td style="padding:5px 8px;font-size:13px;font-weight:600;color:{color};text-align:right;'
                f'border-bottom:1px solid #f3f4f6;">{val_str}</td></tr>'
            )

        pnl_rows: list[str] = []
        if latest_asof:
            pnl_rows.append(_pnl_row("Snapshot", _html_escape(str(latest_asof))))
        if prev_asof:
            pnl_rows.append(_pnl_row("Previous Snapshot", _html_escape(str(prev_asof))))
        pnl_rows.append(_pnl_row("Current Equity (CAD)", f"${_fmt_money(equity)}" if equity is not None else "N/A"))
        if fx_rate is not None and fx_rate > 0 and equity is not None:
            pnl_rows.append(_pnl_row("Current Equity (USD)", f"${_fmt_money(equity / fx_rate)}"))
        pnl_rows.append(_pnl_row("Cash", f"${_fmt_money(cash)}" if cash is not None else "N/A"))
        pnl_rows.append(_pnl_row("Invested Market Value", f"${_fmt_money(open_market_value)}" if open_market_value is not None else "N/A"))
        _rpl_color = "#059669" if realized_pl and realized_pl > 0 else "#dc2626" if realized_pl and realized_pl < 0 else "#111827"
        pnl_rows.append(_pnl_row("Realized P&amp;L", f"${_fmt_money(realized_pl)}" if realized_pl is not None else "N/A", _rpl_color))
        _upl_color = "#059669" if unrealized_pl and unrealized_pl > 0 else "#dc2626" if unrealized_pl and unrealized_pl < 0 else "#111827"
        pnl_rows.append(_pnl_row("Unrealized P&amp;L", f"${_fmt_money(unrealized_pl)}" if unrealized_pl is not None else "N/A", _upl_color))
        _npl_color = "#059669" if net_pl and net_pl > 0 else "#dc2626" if net_pl and net_pl < 0 else "#111827"
        pnl_rows.append(_pnl_row("Net P&amp;L", f"${_fmt_money(net_pl)}" if net_pl is not None else "N/A", _npl_color))

        # Returns section
        _atr_color = "#059669" if all_time_return and all_time_return > 0 else "#dc2626" if all_time_return and all_time_return < 0 else "#111827"
        _dtr_color = "#059669" if day_to_day_return and day_to_day_return > 0 else "#dc2626" if day_to_day_return and day_to_day_return < 0 else "#111827"
        pnl_rows.append(_pnl_row(all_time_label_html, _fmt_pct(all_time_return) if all_time_return is not None else "N/A", _atr_color))
        pnl_rows.append(_pnl_row(day_to_day_label_html, _fmt_pct(day_to_day_return) if day_to_day_return is not None else "N/A", _dtr_color))

        pnl_table_rows = "\n".join(pnl_rows)

        pnl_block = f"""
  <div style="background:#ffffff;border-radius:12px;padding:16px 18px;margin:0 0 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:12px;">Portfolio Returns</div>
    <table style="width:100%;border-collapse:collapse;">
      {pnl_table_rows}
    </table>
  </div>
"""

    # Build target weights HTML table (shown when current holdings are empty)
    target_weights_html_block = ""
    if weights.empty and target_weights is not None and not target_weights.empty:
        tw_table = target_weights.reset_index().copy()
        tw_table = tw_table.rename(columns={tw_table.columns[0]: "ticker"})
        tw_cols = ["ticker", "weight", "score", "last_close_cad", "ret_60d", "vol_60d_ann"]
        tw_cols = [c for c in tw_cols if c in tw_table.columns]
        tw_display = tw_table[tw_cols].copy()
        if "weight" in tw_display.columns:
            tw_display["weight"] = tw_display["weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
        if "last_close_cad" in tw_display.columns:
            if fx_rate is not None and fx_rate > 0:
                tw_display["last_close_usd"] = pd.to_numeric(tw_table["last_close_cad"], errors="coerce") / float(fx_rate)
                tw_display["last_close_usd"] = tw_display["last_close_usd"].map(_fmt_money)
            tw_display["last_close_cad"] = tw_display["last_close_cad"].map(_fmt_money)
        if "score" in tw_display.columns:
            tw_display["score"] = tw_display["score"].map(_fmt_num)
        if "ret_60d" in tw_display.columns:
            tw_display["ret_60d"] = tw_display["ret_60d"].map(_fmt_pct)
        if "vol_60d_ann" in tw_display.columns:
            tw_display["vol_60d_ann"] = tw_display["vol_60d_ann"].map(_fmt_pct)

        tw_headers = "".join(
            f"<th style='text-align:left;padding:6px 8px;border-bottom:2px solid #0ea5e9;'>{_html_escape(c)}</th>"
            for c in tw_display.columns
        )
        tw_rows = "\n".join(
            "<tr>" + "".join(
                f"<td style='padding:6px 8px;border-bottom:1px solid #e5e7eb;'>{_html_escape(str(v))}</td>" for v in row
            ) + "</tr>"
            for row in tw_display.itertuples(index=False, name=None)
        )
        target_weights_html_block = f"""
  <h3 style="margin: 0 0 10px 0; color: #0ea5e9;">Target Portfolio Weights (recommended)</h3>
  <p style="margin:0 0 8px 0;color:#6b7280;font-size:13px;">No current holdings. The model recommends the following positions:</p>
  <table style="border-collapse: collapse; width: 100%; font-size: 13px;">
    <thead><tr>{tw_headers}</tr></thead>
    <tbody>{tw_rows}</tbody>
  </table>
"""

    # ── Risk dashboard + settlement warnings ───────────────────────
    risk_block = ""
    _risk_alerts: list[str] = []
    _risk_segments: list[str] = []  # horizontal bar segments

    # Data freshness
    _is_intraday = run_meta.get("intraday", False) if isinstance(run_meta, dict) else False
    _data_freshness_label = "Intraday 1h bars" if _is_intraday else "Daily close"
    _data_freshness_color = "#f59e0b" if _is_intraday else "#059669"

    # Settlement warning on BUY actions
    _n_buys = sum(1 for a in (trade_actions or []) if (getattr(a, "action", None) or (a.get("action") if isinstance(a, dict) else "")) == "BUY")
    _n_sells = sum(1 for a in (trade_actions or []) if (getattr(a, "action", None) or (a.get("action") if isinstance(a, dict) else "")) in ("SELL", "SELL_PARTIAL"))
    if _n_buys > 0 and _n_sells > 0:
        _risk_alerts.append("T+2 Settlement: Sell proceeds may not settle for 2 business days. Ensure cash is available before placing buy orders.")

    # Drawdown check from P&L history
    _dd_pct = 0.0
    _dd_color = "#059669"
    _dd_label = "0.0%"
    _dd_bar_width = 0
    if portfolio_pnl_history and len(portfolio_pnl_history) >= 2:
        _equities = [float(h.get("equity_cad", 0)) for h in portfolio_pnl_history if h.get("equity_cad")]
        if _equities:
            _peak = max(_equities)
            _current = _equities[-1]
            _dd = (_current / _peak - 1.0) if _peak > 0 else 0
            _dd_pct = _dd
            _dd_label = f"{_dd * 100:.1f}%"
            # Map drawdown to bar width (0% = 0 width, -20% = 100% width)
            _dd_bar_width = min(100, max(0, int(abs(_dd) * 500)))
            if _dd < -0.10:
                _dd_color = "#dc2626"
                _risk_alerts.append(f"SEVERE DRAWDOWN: Portfolio is {_dd*100:.1f}% below peak (${_peak:,.0f} &rarr; ${_current:,.0f}). Review all positions.")
            elif _dd < -0.05:
                _dd_color = "#f59e0b"
                _risk_alerts.append(f"Drawdown alert: Portfolio is {_dd*100:.1f}% below peak equity. Consider reducing position sizes.")
            else:
                _dd_color = "#059669"

    # Kill switch status
    _halt = False
    try:
        import os as _os
        _halt = _os.getenv("TRADING_HALT", "").strip().lower() in ("1", "true", "yes")
    except Exception:
        pass
    if _halt:
        _risk_alerts.append("TRADING HALTED: TRADING_HALT is active. No new trades will be executed.")
    _kill_color = "#dc2626" if _halt else "#059669"
    _kill_label = "HALTED" if _halt else "Active"

    # LLM agent status
    _llm_status = (run_meta.get("llm_agent", {}) or {}).get("status", "disabled") if isinstance(run_meta, dict) else "disabled"

    # Settlement warning indicator
    _settle_color = "#f59e0b" if (_n_buys > 0 and _n_sells > 0) else "#059669"
    _settle_label = "T+2 Warning" if (_n_buys > 0 and _n_sells > 0) else "Clear"

    # Build alerts strip (above the bar)
    _alerts_html = ""
    if _risk_alerts:
        _alerts_html = "".join(
            f'<div style="background:#fef2f2;border-left:4px solid #dc2626;padding:8px 12px;'
            f'margin-bottom:8px;border-radius:0 8px 8px 0;font-size:12px;color:#991b1b;">'
            f'{_html_escape(a)}</div>'
            for a in _risk_alerts
        )

    # Build the horizontal risk dashboard bar
    risk_block = f"""
  <div style="background:#ffffff;border-radius:12px;padding:16px 18px;margin:0 0 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:12px;">Risk Dashboard</div>
    {_alerts_html}
    <table style="width:100%;border-collapse:collapse;"><tr>
      <!-- Drawdown gauge -->
      <td style="vertical-align:top;padding:0 12px 0 0;width:35%;">
        <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;">Drawdown from Peak</div>
        <div style="font-size:16px;font-weight:700;color:{_dd_color};margin-bottom:6px;">{_dd_label}</div>
        <div style="background:#f3f4f6;border-radius:4px;height:8px;overflow:hidden;">
          <div style="background:{_dd_color};width:{_dd_bar_width}%;height:8px;border-radius:4px;"></div>
        </div>
      </td>
      <!-- Data freshness -->
      <td style="vertical-align:top;padding:0 12px;width:20%;">
        <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;">Data Freshness</div>
        <div style="display:inline-block;padding:3px 10px;border-radius:6px;font-size:12px;font-weight:600;
          background:{_data_freshness_color}20;color:{_data_freshness_color};">{_data_freshness_label}</div>
      </td>
      <!-- Kill switch -->
      <td style="vertical-align:top;padding:0 12px;width:20%;">
        <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;">Kill Switch</div>
        <div style="display:inline-block;padding:3px 10px;border-radius:6px;font-size:12px;font-weight:600;
          background:{_kill_color}20;color:{_kill_color};">{_kill_label}</div>
      </td>
      <!-- Settlement -->
      <td style="vertical-align:top;padding:0 0 0 12px;width:25%;">
        <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;">Settlement</div>
        <div style="display:inline-block;padding:3px 10px;border-radius:6px;font-size:12px;font-weight:600;
          background:{_settle_color}20;color:{_settle_color};">{_settle_label}</div>
        <div style="font-size:11px;color:#6b7280;margin-top:4px;">{_n_buys} BUY / {_n_sells} SELL</div>
      </td>
    </tr></table>
  </div>
"""

    total_scanned = total_processed if total_processed is not None else len(screened)
    portfolio_label = f"{len(weights):,} tickers (current holdings)" if not weights.empty else "No current holdings"
    _n_actions = len(trade_actions) if trade_actions else 0

    # ── Executive Summary Cards data ──
    # Equity card
    _eq_display = f"${_fmt_money(_pnl_equity)}" if _pnl_equity is not None else "N/A"
    _eq_change_arrow = ""
    _eq_change_color = "#6b7280"
    _eq_change_str = ""
    if _pnl_day_change is not None:
        _eq_change_arrow = "&#9650;" if _pnl_day_change >= 0 else "&#9660;"
        _eq_change_color = "#059669" if _pnl_day_change >= 0 else "#dc2626"
        _eq_change_str = f"{_pnl_day_change * 100:+.2f}% today"
    _eq_alltime_str = _fmt_pct(_pnl_all_time_return) if _pnl_all_time_return is not None else "N/A"

    # Positions card
    _n_positions = len(weights) if not weights.empty else 0

    # Risk status card (traffic light)
    if _halt:
        _risk_light_color = "#dc2626"
        _risk_light_label = "HALTED"
        _risk_light_bg = "#fef2f2"
    elif _dd_pct < -0.10:
        _risk_light_color = "#dc2626"
        _risk_light_label = "High Risk"
        _risk_light_bg = "#fef2f2"
    elif _dd_pct < -0.05:
        _risk_light_color = "#f59e0b"
        _risk_light_label = "Elevated"
        _risk_light_bg = "#fffbeb"
    else:
        _risk_light_color = "#059669"
        _risk_light_label = "Normal"
        _risk_light_bg = "#ecfdf5"

    # LLM-primary badge
    _llm_primary = run_meta.get("llm_primary", False) if isinstance(run_meta, dict) else False
    _llm_badge = (
        '<span style="background:#8b5cf6;color:#ffffff;font-size:10px;font-weight:700;'
        'padding:3px 10px;border-radius:6px;margin-left:12px;letter-spacing:0.5px;">LLM-Primary</span>'
    ) if _llm_primary else ""

    html = f"""<html>
<body style="font-family:system-ui,-apple-system,Arial,sans-serif;line-height:1.5;color:#111827;
  max-width:900px;margin:0 auto;padding:0;background:#f0f2f5;">

  <!-- ===== 1. DASHBOARD HEADER ===== -->
  <div style="background:#111827;padding:18px 24px;border-radius:0 0 14px 14px;">
    <table style="width:100%;border-collapse:collapse;"><tr>
      <td style="vertical-align:middle;padding:0;">
        <span style="font-size:18px;font-weight:800;color:#ffffff;letter-spacing:-0.3px;">Trading Intelligence Report</span>
        {_llm_badge}
      </td>
      <td style="vertical-align:middle;text-align:right;padding:0;">
        <span style="font-size:12px;color:#9ca3af;">{_html_escape(now)}</span>
      </td>
    </tr></table>
  </div>

  <div style="padding:16px 16px 0 16px;">

  <!-- ===== 2. EXECUTIVE SUMMARY CARDS (3 across) ===== -->
  <table style="width:100%;border-collapse:separate;border-spacing:12px 0;margin:0 0 16px 0;"><tr>
    <!-- Equity Card -->
    <td style="width:33%;vertical-align:top;background:#ffffff;border-radius:12px;padding:16px;
      box-shadow:0 1px 3px rgba(0,0,0,0.08);">
      <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Equity (CAD)</div>
      <div style="font-size:22px;font-weight:800;color:#111827;margin-bottom:4px;">{_eq_display}</div>
      <div style="font-size:13px;color:{_eq_change_color};font-weight:600;">{_eq_change_arrow} {_eq_change_str}</div>
      <div style="font-size:11px;color:#6b7280;margin-top:4px;">All-time: {_eq_alltime_str}</div>
    </td>
    <!-- Positions Card -->
    <td style="width:33%;vertical-align:top;background:#ffffff;border-radius:12px;padding:16px;
      box-shadow:0 1px 3px rgba(0,0,0,0.08);">
      <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Positions</div>
      <div style="font-size:22px;font-weight:800;color:#111827;margin-bottom:4px;">{_n_positions}</div>
      <div style="font-size:13px;color:#6b7280;">open positions</div>
      <div style="font-size:11px;color:#6b7280;margin-top:4px;">{_n_actions} action{"s" if _n_actions != 1 else ""} today</div>
    </td>
    <!-- Risk Status Card -->
    <td style="width:33%;vertical-align:top;background:{_risk_light_bg};border-radius:12px;padding:16px;
      box-shadow:0 1px 3px rgba(0,0,0,0.08);border:2px solid {_risk_light_color}30;">
      <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Risk Status</div>
      <table style="border-collapse:collapse;"><tr>
        <td style="vertical-align:middle;padding:0 8px 0 0;">
          <div style="width:16px;height:16px;border-radius:50%;background:{_risk_light_color};"></div>
        </td>
        <td style="vertical-align:middle;padding:0;">
          <div style="font-size:18px;font-weight:800;color:{_risk_light_color};">{_risk_light_label}</div>
        </td>
      </tr></table>
      <div style="font-size:11px;color:#6b7280;margin-top:6px;">DD: {_dd_label} from peak</div>
    </td>
  </tr></table>

  <!-- ===== 3. ACTION CARDS (open by default) ===== -->
  <div style="background:#ffffff;border-radius:12px;padding:16px 18px;margin:0 0 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <details open>
      <summary style="cursor:pointer;font-size:15px;font-weight:700;color:#111827;margin-bottom:12px;list-style:none;">
        <span style="margin-right:6px;">&#9660;</span> Recommended Actions
      </summary>
      {actions_html}
    </details>
  </div>

  <!-- ===== 4. RISK DASHBOARD (collapsed) ===== -->
  <div style="background:#ffffff;border-radius:12px;padding:16px 18px;margin:0 0 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <details>
      <summary style="cursor:pointer;font-size:15px;font-weight:700;color:#111827;margin-bottom:10px;list-style:none;">
        <span style="margin-right:6px;">&#9654;</span> Risk Dashboard
      </summary>
      <div style="margin-top:10px;">
        {risk_block}
      </div>
    </details>
  </div>

  <!-- ===== 5. LLM AGENT BLOCK (collapsed) ===== -->
  <div style="background:#ffffff;border-radius:12px;padding:16px 18px;margin:0 0 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <details>
      <summary style="cursor:pointer;font-size:15px;font-weight:700;color:#111827;margin-bottom:10px;list-style:none;">
        <span style="margin-right:6px;">&#129302;</span> LLM Agent Analysis
      </summary>
      <div style="margin-top:10px;">
        {llm_block}
      </div>
    </details>
  </div>

  <!-- ===== 6. PORTFOLIO DETAILS ===== -->
  <div style="background:#ffffff;border-radius:12px;padding:16px 18px;margin:0 0 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <details>
      <summary style="cursor:pointer;font-size:15px;font-weight:700;color:#111827;margin-bottom:10px;">
        P&amp;L Breakdown
      </summary>
      <div style="margin-top:10px;">
        {pnl_block}
      </div>
    </details>
  </div>

  {target_weights_html_block}

  <div style="background:#ffffff;border-radius:12px;padding:16px 18px;margin:0 0 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <details>
      <summary style="cursor:pointer;font-size:15px;font-weight:700;color:#111827;">
        Current Portfolio Holdings ({portfolio_label})
      </summary>
      <div style="margin-top:12px;overflow-x:auto;">
        <table style="border-collapse:collapse;width:100%;font-size:12px;">
          <thead>
            <tr style="background:#f9fafb;">{headers_html}</tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </details>
  </div>

  <div style="background:#ffffff;border-radius:12px;padding:16px 18px;margin:0 0 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <details>
      <summary style="cursor:pointer;font-size:15px;font-weight:700;color:#111827;">
        Universe &amp; Screening Stats
      </summary>
      <div style="margin-top:10px;">
        <table style="border-collapse:collapse;width:100%;font-size:12px;">
          <tr><td style="padding:5px 8px;color:#6b7280;border-bottom:1px solid #f3f4f6;">Universe</td>
              <td style="padding:5px 8px;font-weight:600;border-bottom:1px solid #f3f4f6;">US + TSX</td></tr>
          <tr><td style="padding:5px 8px;color:#6b7280;border-bottom:1px solid #f3f4f6;">Tickers Scanned</td>
              <td style="padding:5px 8px;font-weight:600;border-bottom:1px solid #f3f4f6;">{total_scanned:,}</td></tr>
          <tr><td style="padding:5px 8px;color:#6b7280;border-bottom:1px solid #f3f4f6;">Top Screened</td>
              <td style="padding:5px 8px;font-weight:600;border-bottom:1px solid #f3f4f6;">{len(screened):,} tickers</td></tr>
          <tr><td style="padding:5px 8px;color:#6b7280;">Portfolio</td>
              <td style="padding:5px 8px;font-weight:600;">{portfolio_label}</td></tr>
        </table>
      </div>
    </details>
  </div>

  <div style="background:#ffffff;border-radius:12px;padding:16px 18px;margin:0 0 16px 0;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <details>
      <summary style="cursor:pointer;font-size:15px;font-weight:700;color:#111827;">
        Model Validation
      </summary>
      <div style="margin-top:10px;">
        {model_block if model_block else '<div style="font-size:12px;color:#9ca3af;">No model validation data available.</div>'}
      </div>
    </details>
  </div>

  <div style="text-align:center;padding:16px 0 8px 0;font-size:11px;color:#9ca3af;">
    Attachments: daily_report.txt (full details) &bull; portfolio_weights.csv (weights + metrics)
  </div>

  </div><!-- /padding wrapper -->
</body>
</html>"""

    (reports_dir / "daily_email.html").write_text(html, encoding="utf-8")
    if trade_actions:
        # Persist as JSON for debugging/auditing.
        try:
            import json

            payload = []
            for a in trade_actions:
                if isinstance(a, dict):
                    payload.append(a)
                else:
                    px = getattr(a, "price_cad", None)
                    pred_ret = getattr(a, "pred_return", None)
                    sell_px = float(px) * (1 + float(pred_ret)) if px and pred_ret is not None else None
                    payload.append(
                        {
                            "ticker": getattr(a, "ticker", None),
                            "action": getattr(a, "action", None),
                            "reason": getattr(a, "reason", None),
                            "shares": getattr(a, "shares", None),
                            "price_cad": px,
                            "days_held": getattr(a, "days_held", None),
                            "pred_return": pred_ret,
                            "sell_price_cad": sell_px,
                            "expected_sell_date": getattr(a, "expected_sell_date", None),
                            "replaces_ticker": getattr(a, "replaces_ticker", None),
                        }
                    )
            (reports_dir / "trade_actions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not write trade_actions.json: %s", e)
    logger.info("Rendered reports: %s", str(reports_dir))


def render_intraday_report(
    reports_dir: Path,
    positions: list[dict],
    exit_actions: list,
    run_meta: dict,
    logger,
    *,
    entry_actions: list | None = None,
) -> None:
    """Render a lightweight intraday monitoring report."""
    from datetime import datetime, timezone
    reports_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    status = run_meta.get("status", "unknown")

    # --- Text report ---
    lines = [
        "=" * 60,
        "INTRADAY PORTFOLIO MONITOR",
        "=" * 60,
        f"Generated: {now}",
        f"Status: {status}",
        "",
    ]

    if positions:
        lines.append("OPEN POSITIONS")
        lines.append("-" * 60)
        lines.append(f"{'Ticker':<12} {'Entry':>10} {'Current':>10} {'P&L':>8} {'Days':>5}")
        for p in positions:
            ticker = p.get("ticker", "")
            entry = p.get("entry_price")
            current = p.get("current_price")
            pnl = p.get("pnl_pct")
            days = p.get("days_held", "")
            entry_str = _fmt_money(entry) if entry else "N/A"
            current_str = _fmt_money(current) if current else "N/A"
            pnl_str = _fmt_pct(pnl) if pnl is not None else "N/A"
            lines.append(f"{ticker:<12} {entry_str:>10} {current_str:>10} {pnl_str:>8} {days:>5}")
        lines.append("")

    if exit_actions:
        lines.append("EXIT ACTIONS")
        lines.append("-" * 60)
        for a in exit_actions:
            if isinstance(a, dict):
                ticker = a.get("ticker", "")
                reason = a.get("reason", "")
            else:
                ticker = getattr(a, "ticker", "")
                reason = getattr(a, "reason", "")
            lines.append(f"  SELL {ticker} — {reason}")
        lines.append("")

    if entry_actions:
        lines.append("ENTRY ACTIONS")
        lines.append("-" * 60)
        for a in entry_actions:
            if isinstance(a, dict):
                ticker = a.get("ticker", "")
                shares = a.get("shares", "")
                px = a.get("price_cad")
            else:
                ticker = getattr(a, "ticker", "")
                shares = getattr(a, "shares", "")
                px = getattr(a, "price_cad", None)
            px_str = _fmt_money(px) if px else "N/A"
            lines.append(f"  BUY  {ticker} — {shares} shares @ {px_str}")
        lines.append("")

    if not exit_actions and not entry_actions:
        lines.append("No actions triggered.")
        lines.append("")

    (reports_dir / "intraday_report.txt").write_text("\n".join(lines), encoding="utf-8")

    # --- HTML email ---
    position_rows = ""
    for p in positions:
        ticker = p.get("ticker", "")
        entry = p.get("entry_price")
        current = p.get("current_price")
        pnl = p.get("pnl_pct")
        days = p.get("days_held", "")
        trailing = p.get("trailing_stop")
        pnl_val = float(pnl) if pnl is not None and pnl == pnl else None
        color = "#059669" if pnl_val and pnl_val > 0 else "#dc2626" if pnl_val and pnl_val < 0 else "#666"
        position_rows += (
            f"<tr>"
            f"<td style='padding:6px 8px;font-weight:bold;'>{_html_escape(str(ticker))}</td>"
            f"<td style='padding:6px 8px;'>{_fmt_money(entry) if entry else 'N/A'}</td>"
            f"<td style='padding:6px 8px;'>{_fmt_money(current) if current else 'N/A'}</td>"
            f"<td style='padding:6px 8px;color:{color};font-weight:bold;'>{_fmt_pct(pnl) if pnl is not None else 'N/A'}</td>"
            f"<td style='padding:6px 8px;'>{days}</td>"
            f"<td style='padding:6px 8px;'>{_fmt_money(trailing) if trailing else 'N/A'}</td>"
            f"</tr>"
        )

    exit_html = ""
    if exit_actions:
        exit_rows = ""
        for a in exit_actions:
            if isinstance(a, dict):
                ticker = a.get("ticker", "")
                reason = a.get("reason", "")
                px = a.get("price_cad")
            else:
                ticker = getattr(a, "ticker", "")
                reason = getattr(a, "reason", "")
                px = getattr(a, "price_cad", None)
            exit_rows += (
                f"<tr>"
                f"<td style='padding:4px 8px;color:#dc2626;font-weight:bold;'>SELL</td>"
                f"<td style='padding:4px 8px;font-weight:bold;'>{_html_escape(str(ticker))}</td>"
                f"<td style='padding:4px 8px;'>{_fmt_money(px) if px else 'N/A'}</td>"
                f"<td style='padding:4px 8px;'>{_html_escape(str(reason))}</td>"
                f"</tr>"
            )
        exit_html = f"""
        <h3 style="color:#dc2626;margin:16px 0 8px 0;">Exit Actions Triggered</h3>
        <table style="border-collapse:collapse;width:100%;font-size:13px;">
        <thead><tr>
            <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #dc2626;">Action</th>
            <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #dc2626;">Ticker</th>
            <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #dc2626;">Price</th>
            <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #dc2626;">Reason</th>
        </tr></thead>
        <tbody>{exit_rows}</tbody>
        </table>
        """

    entry_html = ""
    if entry_actions:
        entry_rows = ""
        for a in entry_actions:
            if isinstance(a, dict):
                ticker = a.get("ticker", "")
                shares = a.get("shares", "")
                px = a.get("price_cad")
            else:
                ticker = getattr(a, "ticker", "")
                shares = getattr(a, "shares", "")
                px = getattr(a, "price_cad", None)
            entry_rows += (
                f"<tr>"
                f"<td style='padding:4px 8px;color:#059669;font-weight:bold;'>BUY</td>"
                f"<td style='padding:4px 8px;font-weight:bold;'>{_html_escape(str(ticker))}</td>"
                f"<td style='padding:4px 8px;'>{shares}</td>"
                f"<td style='padding:4px 8px;'>{_fmt_money(px) if px else 'N/A'}</td>"
                f"</tr>"
            )
        entry_html = f"""
        <h3 style="color:#059669;margin:16px 0 8px 0;">Entry Actions</h3>
        <table style="border-collapse:collapse;width:100%;font-size:13px;">
        <thead><tr>
            <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #059669;">Action</th>
            <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #059669;">Ticker</th>
            <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #059669;">Shares</th>
            <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #059669;">Price</th>
        </tr></thead>
        <tbody>{entry_rows}</tbody>
        </table>
        """

    n_exits = run_meta.get("n_exits", 0)
    n_entries = run_meta.get("n_entries", 0)
    max_move = run_meta.get("max_move_pct", 0)
    entries_enabled = run_meta.get("entries_enabled", False)
    status_message = run_meta.get("message", "")
    status_badge = {
        "executed": "Exit logic evaluated" + (" + entries" if entries_enabled else ""),
        "below_threshold": "Below move threshold — no action",
        "no_positions": "No open positions" + (" — checking entries" if entries_enabled else ""),
        "no_intraday_data": "Market closed — no data",
        "no_cache": "Waiting for daily post-close run to populate cache",
        "no_daily_run": "No daily run found — run daily pipeline first",
        "stale": "Daily signal too old — waiting for next post-close run",
        "no_tickers": "No tickers to scan",
        "no_prices": "Could not fetch prices",
    }.get(status, status)
    if status_message and status not in ("executed",):
        status_badge += f" — {_html_escape(str(status_message))}"

    html = f"""<html>
<body style="font-family:Arial,sans-serif;line-height:1.5;color:#111827;max-width:700px;margin:0 auto;padding:20px;">
  <h2 style="margin:0 0 8px 0;">Intraday Portfolio Monitor</h2>
  <p style="margin:0 0 12px 0;color:#6b7280;font-size:13px;">{_html_escape(now)} &mdash; {_html_escape(status_badge)}</p>

  {exit_html}

  {entry_html}

  <h3 style="margin:16px 0 8px 0;">Open Positions</h3>
  <table style="border-collapse:collapse;width:100%;font-size:13px;">
  <thead><tr>
      <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Ticker</th>
      <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Entry</th>
      <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Current</th>
      <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">P&amp;L</th>
      <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Days</th>
      <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Trail Stop</th>
  </tr></thead>
  <tbody>{position_rows if position_rows else '<tr><td colspan="6" style="padding:8px;color:#9ca3af;">No open positions</td></tr>'}</tbody>
  </table>

  <p style="margin-top:16px;color:#6b7280;font-size:12px;">
    Max price move: {max_move * 100:.1f}% &bull; Exits: {n_exits} &bull; Entries: {n_entries} &bull; Next full run: post-close
  </p>
</body>
</html>"""

    (reports_dir / "intraday_email.html").write_text(html, encoding="utf-8")
    logger.info("Rendered intraday report: %s", str(reports_dir))

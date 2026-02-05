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


def render_reports(
    reports_dir: Path,
    run_meta: dict[str, Any],
    universe_meta: dict[str, Any],
    screened: pd.DataFrame,
    weights: pd.DataFrame,
    trade_actions: list[Any] | None,
    logger,
    *,
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

    lines.append("PORTFOLIO WEIGHTS (inverse-vol, capped)")
    lines.append("-" * 78)
    weights_view = weights.copy()
    fx_rate = _to_float(fx_usdcad_rate)
    if fx_rate is not None and fx_rate > 0 and "last_close_cad" in weights_view.columns:
        weights_view["last_close_usd"] = pd.to_numeric(weights_view["last_close_cad"], errors="coerce") / float(fx_rate)
    else:
        weights_view["last_close_usd"] = pd.NA
    if "shares" in weights_view.columns and "last_close_cad" in weights_view.columns:
        weights_view["position_value_cad"] = pd.to_numeric(weights_view["last_close_cad"], errors="coerce") * pd.to_numeric(
            weights_view["shares"], errors="coerce"
        )
    else:
        weights_view["position_value_cad"] = pd.NA
    if fx_rate is not None and fx_rate > 0:
        weights_view["position_value_usd"] = pd.to_numeric(weights_view["position_value_cad"], errors="coerce") / float(fx_rate)
    else:
        weights_view["position_value_usd"] = pd.NA
    weights_view["weight"] = weights_view["weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
    weights_view["last_close_cad"] = weights_view["last_close_cad"].map(_fmt_money)
    weights_view["last_close_usd"] = weights_view["last_close_usd"].map(_fmt_money)
    if "shares" in weights_view.columns:
        weights_view["shares"] = weights_view["shares"].map(lambda x: str(int(x)) if _to_float(x) is not None else "N/A")
    else:
        weights_view["shares"] = pd.NA
    weights_view["position_value_cad"] = weights_view["position_value_cad"].map(_fmt_money)
    weights_view["position_value_usd"] = weights_view["position_value_usd"].map(_fmt_money)
    weights_view["ret_60d"] = weights_view["ret_60d"].map(_fmt_pct)
    weights_view["vol_60d_ann"] = weights_view["vol_60d_ann"].map(_fmt_pct)
    weights_cols = [
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
    lines.extend(weights_view[weights_cols].to_string().splitlines())
    lines.append("")

    if trade_actions:
        lines.append("RECOMMENDED ACTIONS (max hold)")
        lines.append("-" * 78)
        for a in trade_actions:
            # Support both dataclass actions and dict-like actions.
            ticker = getattr(a, "ticker", None) or (a.get("ticker") if isinstance(a, dict) else "")
            action = getattr(a, "action", None) or (a.get("action") if isinstance(a, dict) else "")
            reason = getattr(a, "reason", None) or (a.get("reason") if isinstance(a, dict) else "")
            shares = getattr(a, "shares", None) or (a.get("shares") if isinstance(a, dict) else "")
            px = getattr(a, "price_cad", None) or (a.get("price_cad") if isinstance(a, dict) else "")
            days = getattr(a, "days_held", None) or (a.get("days_held") if isinstance(a, dict) else "")
            pred_ret = getattr(a, "pred_return", None) or (a.get("pred_return") if isinstance(a, dict) else None)
            sell_date = getattr(a, "expected_sell_date", None) or (a.get("expected_sell_date") if isinstance(a, dict) else "")
            pred_ret_str = _fmt_pct(pred_ret) if pred_ret is not None else "N/A"
            px_cad_str = _fmt_money(px) if px else "N/A"
            px_usd_str = _fmt_money(float(px) / fx_rate) if px and fx_rate and fx_rate > 0 else "N/A"
            # Calculate target sell price based on predicted return
            sell_px_cad = float(px) * (1 + float(pred_ret)) if px and pred_ret is not None else None
            sell_px_cad_str = _fmt_money(sell_px_cad) if sell_px_cad is not None else "N/A"
            sell_px_usd_str = _fmt_money(sell_px_cad / fx_rate) if sell_px_cad and fx_rate and fx_rate > 0 else "N/A"
            lines.append(f"{action:>4} {ticker:<12} shares={shares} price_cad={px_cad_str} price_usd={px_usd_str} days_held={days} pred_ret={pred_ret_str} sell_price_cad={sell_px_cad_str} sell_price_usd={sell_px_usd_str} sell_date={sell_date or 'N/A'} reason={reason}")
        lines.append("")

    # Portfolio P&L history (stateful; computed from portfolio state positions)
    if portfolio_pnl_history:
        lines.append("PORTFOLIO RETURNS")
        lines.append("-" * 78)
        latest = portfolio_pnl_history[-1]
        prev = portfolio_pnl_history[-2] if len(portfolio_pnl_history) >= 2 else None
        first = portfolio_pnl_history[0]
        
        equity = _to_float(latest.get("equity_cad"))
        prev_equity = _to_float(prev.get("equity_cad")) if prev else None
        first_equity = _to_float(first.get("equity_cad"))
        
        # Calculate returns
        all_time_return = None
        day_to_day_return = None
        
        if equity is not None and first_equity is not None and first_equity > 0:
            all_time_return = (equity - first_equity) / first_equity
        
        if equity is not None and prev_equity is not None and prev_equity > 0:
            day_to_day_return = (equity - prev_equity) / prev_equity
        
        lines.append(f"Current Equity: {_fmt_money(equity) if equity is not None else 'N/A'}")
        if fx_rate is not None and fx_rate > 0 and equity is not None:
            lines.append(f"Current Equity USD: {_fmt_money(equity / fx_rate)}")
        lines.append("")
        lines.append(f"All-Time Return: {_fmt_pct(all_time_return) if all_time_return is not None else 'N/A'}")
        lines.append(f"Day-to-Day Return: {_fmt_pct(day_to_day_return) if day_to_day_return is not None else 'N/A'}")
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
    def _html_escape(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    weights_table = weights.reset_index()[["ticker", "weight", "score", "last_close_cad", "ret_60d", "vol_60d_ann"]].copy()
    fx_rate = _to_float(fx_usdcad_rate)
    if fx_rate is not None and fx_rate > 0:
        weights_table["last_close_usd"] = pd.to_numeric(weights_table["last_close_cad"], errors="coerce") / float(fx_rate)
    else:
        weights_table["last_close_usd"] = pd.NA
    if "shares" in weights.columns:
        shares = weights["shares"].copy()
        shares.index = weights.index.astype(str)
        weights_table["shares"] = weights_table["ticker"].astype(str).map(shares)
    else:
        weights_table["shares"] = pd.NA
    weights_table["position_value_cad"] = pd.to_numeric(weights_table["last_close_cad"], errors="coerce") * pd.to_numeric(
        weights_table["shares"], errors="coerce"
    )
    if fx_rate is not None and fx_rate > 0:
        weights_table["position_value_usd"] = pd.to_numeric(weights_table["position_value_cad"], errors="coerce") / float(fx_rate)
    else:
        weights_table["position_value_usd"] = pd.NA
    weights_table["weight"] = weights_table["weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
    weights_table["last_close_cad"] = weights_table["last_close_cad"].map(_fmt_money)
    weights_table["last_close_usd"] = weights_table["last_close_usd"].map(_fmt_money)
    weights_table["shares"] = weights_table["shares"].map(lambda x: str(int(x)) if _to_float(x) is not None else "N/A")
    weights_table["position_value_cad"] = weights_table["position_value_cad"].map(_fmt_money)
    weights_table["position_value_usd"] = weights_table["position_value_usd"].map(_fmt_money)
    weights_table["ret_60d"] = weights_table["ret_60d"].map(_fmt_pct)
    weights_table["vol_60d_ann"] = weights_table["vol_60d_ann"].map(_fmt_pct)
    weights_table["score"] = weights_table["score"].map(_fmt_num)
    weights_table = weights_table[
        [
            "ticker",
            "weight",
            "shares",
            "position_value_cad",
            "position_value_usd",
            "score",
            "last_close_cad",
            "last_close_usd",
            "ret_60d",
            "vol_60d_ann",
        ]
    ].copy()

    rows_html = "\n".join(
        "<tr>"
        + "".join(f"<td style='padding:6px 8px;border-bottom:1px solid #e5e7eb;'>{_html_escape(str(v))}</td>" for v in row)
        + "</tr>"
        for row in weights_table.itertuples(index=False, name=None)
    )

    # Build actions block outside the f-string to avoid complex nested expressions.
    fx_rate = _to_float(fx_usdcad_rate)
    if trade_actions:
        # Build actions as a table for better readability
        action_rows: list[str] = []
        for a in trade_actions:
            ticker = getattr(a, "ticker", None) or (a.get("ticker") if isinstance(a, dict) else "")
            action = getattr(a, "action", None) or (a.get("action") if isinstance(a, dict) else "")
            reason = getattr(a, "reason", None) or (a.get("reason") if isinstance(a, dict) else "")
            shares = getattr(a, "shares", None) or (a.get("shares") if isinstance(a, dict) else "")
            px = getattr(a, "price_cad", None) or (a.get("price_cad") if isinstance(a, dict) else "")
            pred_ret = getattr(a, "pred_return", None) or (a.get("pred_return") if isinstance(a, dict) else None)
            sell_date = getattr(a, "expected_sell_date", None) or (a.get("expected_sell_date") if isinstance(a, dict) else "")
            pred_ret_str = _fmt_pct(pred_ret) if pred_ret is not None else "N/A"
            px_cad_str = _fmt_money(px) if px else "N/A"
            px_usd_str = _fmt_money(float(px) / fx_rate) if px and fx_rate and fx_rate > 0 else "N/A"
            # Calculate target sell price based on predicted return
            sell_px_cad = float(px) * (1 + float(pred_ret)) if px and pred_ret is not None else None
            sell_px_cad_str = _fmt_money(sell_px_cad) if sell_px_cad is not None else "N/A"
            sell_px_usd_str = _fmt_money(sell_px_cad / fx_rate) if sell_px_cad and fx_rate and fx_rate > 0 else "N/A"
            action_rows.append(
                f"<tr><td style='padding:4px 8px;'>{_html_escape(action)}</td>"
                f"<td style='padding:4px 8px;font-weight:bold;'>{_html_escape(str(ticker))}</td>"
                f"<td style='padding:4px 8px;'>{shares}</td>"
                f"<td style='padding:4px 8px;'>{px_cad_str}</td>"
                f"<td style='padding:4px 8px;'>{px_usd_str}</td>"
                f"<td style='padding:4px 8px;'>{pred_ret_str}</td>"
                f"<td style='padding:4px 8px;color:#059669;font-weight:bold;'>{sell_px_cad_str}</td>"
                f"<td style='padding:4px 8px;color:#059669;'>{sell_px_usd_str}</td>"
                f"<td style='padding:4px 8px;'>{sell_date or 'N/A'}</td>"
                f"<td style='padding:4px 8px;'>{_html_escape(str(reason))}</td></tr>"
            )
        if action_rows:
            actions_html = f"""<table style="border-collapse:collapse;width:100%;font-size:13px;">
            <thead><tr>
                <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #d97706;">Action</th>
                <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #d97706;">Ticker</th>
                <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #d97706;">Shares</th>
                <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #d97706;">Price (CAD)</th>
                <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #d97706;">Price (USD)</th>
                <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #d97706;">Pred Ret</th>
                <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #059669;color:#059669;">Sell @ (CAD)</th>
                <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #059669;color:#059669;">Sell @ (USD)</th>
                <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #d97706;">Sell Date</th>
                <th style="text-align:left;padding:4px 8px;border-bottom:1px solid #d97706;">Reason</th>
            </tr></thead>
            <tbody>{"".join(action_rows)}</tbody>
            </table>"""
        else:
            actions_html = _html_escape("No actions (portfolio already aligned).")
    else:
        actions_html = _html_escape("No actions (portfolio already aligned).")

    pnl_block = ""
    if portfolio_pnl_history:
        latest = portfolio_pnl_history[-1]
        prev = portfolio_pnl_history[-2] if len(portfolio_pnl_history) >= 2 else None
        first = portfolio_pnl_history[0]
        
        equity = _to_float(latest.get("equity_cad"))
        prev_equity = _to_float(prev.get("equity_cad")) if prev else None
        first_equity = _to_float(first.get("equity_cad"))
        
        # Calculate returns
        all_time_return = None
        day_to_day_return = None
        
        if equity is not None and first_equity is not None and first_equity > 0:
            all_time_return = (equity - first_equity) / first_equity
        
        if equity is not None and prev_equity is not None and prev_equity > 0:
            day_to_day_return = (equity - prev_equity) / prev_equity
        
        fx_rate = _to_float(fx_usdcad_rate)
        equity_parts: list[str] = []
        equity_parts.append(f"<strong>Current Equity:</strong> {_fmt_money(equity) if equity is not None else 'N/A'}")
        if fx_rate is not None and fx_rate > 0 and equity is not None:
            equity_parts.append(f"<strong>Current Equity USD:</strong> {_fmt_money(equity / fx_rate)}")
        
        return_parts: list[str] = []
        return_parts.append(f"<strong>All-Time Return:</strong> {_fmt_pct(all_time_return) if all_time_return is not None else 'N/A'}")
        return_parts.append(f"<strong>Day-to-Day Return:</strong> {_fmt_pct(day_to_day_return) if day_to_day_return is not None else 'N/A'}")
        
        summary = "<br/>".join(equity_parts + return_parts)

        pnl_block = f"""
  <h3 style="margin: 0 0 10px 0;">Portfolio Returns</h3>
  <div style="background:#ecfeff;border-radius:8px;padding:12px 14px;margin: 0 0 18px 0;">
    <div>{summary}</div>
  </div>
"""

    total_scanned = total_processed if total_processed is not None else len(screened)
    html = f"""<html>
<body style="font-family: Arial, sans-serif; line-height: 1.5; color: #111827; max-width: 900px; margin: 0 auto; padding: 20px;">
  <h2 style="margin: 0 0 10px 0;">Daily Screener + Risk Parity Portfolio (CAD)</h2>
  <p style="margin: 0 0 16px 0; color: #374151;">
    Generated: <strong>{_html_escape(now)}</strong>
  </p>

  <h3 style="margin: 0 0 10px 0;">Recommended Actions (max hold)</h3>
  <div style="background:#fef3c7;border-radius:8px;padding:12px 14px;margin: 0 0 18px 0;">
    {actions_html}
  </div>

  <div style="background:#f3f4f6;border-radius:8px;padding:12px 14px;margin: 0 0 18px 0;">
    <div><strong>Universe:</strong> US + TSX</div>
    <div><strong>Number of tickers scanned:</strong> {total_scanned:,}</div>
    <div><strong>Top screened:</strong> {len(screened):,} tickers</div>
    <div><strong>Portfolio:</strong> {len(weights):,} tickers (inverse-vol weights)</div>
  </div>

  {model_block}

  {pnl_block}

  <h3 style="margin: 0 0 10px 0;">Recommended Portfolio Weights</h3>
  <table style="border-collapse: collapse; width: 100%; font-size: 13px;">
    <thead>
      <tr>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Ticker</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Weight</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Shares</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Value (CAD)</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Value (USD)</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Score</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Price (CAD)</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Price (USD)</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Ret 60d</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Vol 60d (ann)</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <p style="margin-top: 16px; color: #374151; font-size: 13px;">
    Attachments: <strong>daily_report.txt</strong> (full details), <strong>portfolio_weights.csv</strong> (weights + metrics).
  </p>
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
                        }
                    )
            (reports_dir / "trade_actions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not write trade_actions.json: %s", e)
    logger.info("Rendered reports: %s", str(reports_dir))



from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt_money(x: float) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "N/A"


def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x) * 100.0:+.2f}%"
    except Exception:
        return "N/A"


def _fmt_num(x: float) -> str:
    try:
        return f"{float(x):,.3f}"
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
    lines.append("")

    def _to_float(x: Any) -> float | None:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

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
            lines.append(f"{action:>4} {ticker:<12} shares={shares} price_cad={px} days_held={days} reason={reason}")
        lines.append("")

    # Portfolio P&L history (stateful; computed from portfolio state positions)
    if portfolio_pnl_history:
        lines.append("PORTFOLIO NET P/L (CAD; based on `shares` in portfolio state)")
        lines.append("-" * 78)
        latest = portfolio_pnl_history[-1]
        prev = portfolio_pnl_history[-2] if len(portfolio_pnl_history) >= 2 else None
        net = _to_float(latest.get("net_pl_cad"))
        realized = _to_float(latest.get("realized_pl_cad"))
        unrealized = _to_float(latest.get("unrealized_pl_cad"))
        mv = _to_float(latest.get("open_market_value_cad"))
        cash = _to_float(latest.get("cash_cad"))
        equity = _to_float(latest.get("equity_cad"))

        delta_1d = None
        if prev is not None:
            prev_net = _to_float(prev.get("net_pl_cad"))
            if prev_net is not None and net is not None:
                delta_1d = net - prev_net

        parts: list[str] = []
        if equity is not None:
            parts.append(f"Equity: {_fmt_money(equity)}")
            if fx_rate is not None and fx_rate > 0:
                parts.append(f"Equity USD: {_fmt_money(equity / fx_rate)}")
        if cash is not None:
            parts.append(f"Cash: {_fmt_money(cash)}")
        if net is not None:
            parts.append(f"Net: {_fmt_money(net)}")
        if delta_1d is not None:
            parts.append(f"Δ1D: {_fmt_money(delta_1d)}")
        if realized is not None:
            parts.append(f"Realized: {_fmt_money(realized)}")
        if unrealized is not None:
            parts.append(f"Unrealized: {_fmt_money(unrealized)}")
        if mv is not None:
            parts.append(f"Open MV: {_fmt_money(mv)}")
        if parts:
            lines.append(" | ".join(parts))
            lines.append("")

        # Show last 10 points (most recent last)
        tail = portfolio_pnl_history[-10:]
        for item in tail:
            asof = str(item.get("asof_utc", "")).strip()
            day = asof[:10] if len(asof) >= 10 else asof
            item_net = _to_float(item.get("net_pl_cad"))
            item_real = _to_float(item.get("realized_pl_cad"))
            item_unreal = _to_float(item.get("unrealized_pl_cad"))
            lines.append(
                f"{day:<12} net={_fmt_money(item_net) if item_net is not None else 'N/A'} "
                f"realized={_fmt_money(item_real) if item_real is not None else 'N/A'} "
                f"unrealized={_fmt_money(item_unreal) if item_unreal is not None else 'N/A'}"
            )
        lines.append("")

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
    if trade_actions:
        parts: list[str] = []
        for a in trade_actions:
            ticker = getattr(a, "ticker", None) or (a.get("ticker") if isinstance(a, dict) else "")
            action = getattr(a, "action", None) or (a.get("action") if isinstance(a, dict) else "")
            reason = getattr(a, "reason", None) or (a.get("reason") if isinstance(a, dict) else "")
            parts.append(_html_escape(f"{action} {ticker} ({reason})"))
        actions_html = "<br/>".join(parts) if parts else _html_escape("No actions (portfolio already aligned).")
    else:
        actions_html = _html_escape("No actions (portfolio already aligned).")

    pnl_block = ""
    if portfolio_pnl_history:
        latest = portfolio_pnl_history[-1]
        prev = portfolio_pnl_history[-2] if len(portfolio_pnl_history) >= 2 else None
        net = _to_float(latest.get("net_pl_cad"))
        realized = _to_float(latest.get("realized_pl_cad"))
        unrealized = _to_float(latest.get("unrealized_pl_cad"))
        mv = _to_float(latest.get("open_market_value_cad"))
        cash = _to_float(latest.get("cash_cad"))
        equity = _to_float(latest.get("equity_cad"))
        delta_1d = None
        if prev is not None:
            prev_net = _to_float(prev.get("net_pl_cad"))
            if prev_net is not None and net is not None:
                delta_1d = net - prev_net

        summary_parts: list[str] = []
        if equity is not None:
            summary_parts.append(f"Equity: {_fmt_money(equity)}")
            fx_rate = _to_float(fx_usdcad_rate)
            if fx_rate is not None and fx_rate > 0:
                summary_parts.append(f"Equity USD: {_fmt_money(equity / fx_rate)}")
        if cash is not None:
            summary_parts.append(f"Cash: {_fmt_money(cash)}")
        if net is not None:
            summary_parts.append(f"Net: {_fmt_money(net)}")
        if delta_1d is not None:
            summary_parts.append(f"Δ1D: {_fmt_money(delta_1d)}")
        if realized is not None:
            summary_parts.append(f"Realized: {_fmt_money(realized)}")
        if unrealized is not None:
            summary_parts.append(f"Unrealized: {_fmt_money(unrealized)}")
        if mv is not None:
            summary_parts.append(f"Open MV: {_fmt_money(mv)}")
        summary = " | ".join(summary_parts) if summary_parts else "N/A"

        tail = portfolio_pnl_history[-10:]
        pnl_rows = "\n".join(
            "<tr>"
            + f"<td style='padding:6px 8px;border-bottom:1px solid #e5e7eb;'>{_html_escape(str(item.get('asof_utc',''))[:10])}</td>"
            + f"<td style='padding:6px 8px;border-bottom:1px solid #e5e7eb;'>{_html_escape(_fmt_money(item.get('net_pl_cad')))}</td>"
            + f"<td style='padding:6px 8px;border-bottom:1px solid #e5e7eb;'>{_html_escape(_fmt_money(item.get('realized_pl_cad')))}</td>"
            + f"<td style='padding:6px 8px;border-bottom:1px solid #e5e7eb;'>{_html_escape(_fmt_money(item.get('unrealized_pl_cad')))}</td>"
            + "</tr>"
            for item in tail
        )

        pnl_block = f"""
  <h3 style="margin: 0 0 10px 0;">Portfolio Net P&amp;L (CAD)</h3>
  <div style="background:#ecfeff;border-radius:8px;padding:12px 14px;margin: 0 0 18px 0;">
    <div><strong>{_html_escape(summary)}</strong></div>
    <div style="margin-top:10px; font-size: 12px; color:#374151;">Last {len(tail)} points (UTC date)</div>
    <table style="border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px;">
      <thead>
        <tr>
          <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">As of</th>
          <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Net</th>
          <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Realized</th>
          <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Unrealized</th>
        </tr>
      </thead>
      <tbody>
        {pnl_rows}
      </tbody>
    </table>
    <div style="margin-top:10px; font-size: 12px; color:#374151;">
      Note: computed from entry/exit prices in the persisted portfolio state (uses each position's <code>shares</code>).
    </div>
  </div>
"""

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
    <div><strong>Screened:</strong> {len(screened):,} tickers</div>
    <div><strong>Portfolio:</strong> {len(weights):,} tickers (inverse-vol weights)</div>
  </div>

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
                    payload.append(
                        {
                            "ticker": getattr(a, "ticker", None),
                            "action": getattr(a, "action", None),
                            "reason": getattr(a, "reason", None),
                            "shares": getattr(a, "shares", None),
                            "price_cad": getattr(a, "price_cad", None),
                            "days_held": getattr(a, "days_held", None),
                        }
                    )
            (reports_dir / "trade_actions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not write trade_actions.json: %s", e)
    logger.info("Rendered reports: %s", str(reports_dir))



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
    weights_view["weight"] = weights_view["weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
    weights_view["last_close_cad"] = weights_view["last_close_cad"].map(_fmt_money)
    weights_view["ret_60d"] = weights_view["ret_60d"].map(_fmt_pct)
    weights_view["vol_60d_ann"] = weights_view["vol_60d_ann"].map(_fmt_pct)
    weights_cols = ["weight", "score", "last_close_cad", "ret_60d", "vol_60d_ann", "avg_dollar_volume_cad"]
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
    weights_table["weight"] = weights_table["weight"].map(lambda x: _fmt_pct(x).replace("+", ""))
    weights_table["last_close_cad"] = weights_table["last_close_cad"].map(_fmt_money)
    weights_table["ret_60d"] = weights_table["ret_60d"].map(_fmt_pct)
    weights_table["vol_60d_ann"] = weights_table["vol_60d_ann"].map(_fmt_pct)
    weights_table["score"] = weights_table["score"].map(_fmt_num)

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

  <h3 style="margin: 0 0 10px 0;">Recommended Portfolio Weights</h3>
  <table style="border-collapse: collapse; width: 100%; font-size: 13px;">
    <thead>
      <tr>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Ticker</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Weight</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Score</th>
        <th style="text-align:left;padding:6px 8px;border-bottom:2px solid #111827;">Price (CAD)</th>
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



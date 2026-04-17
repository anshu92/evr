"""Macro insights + portfolio HTML/text/CSV/JSON reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stock_screener.macro.prices import fetch_usdcad_last, last_close_cad
from stock_screener.reporting.email_style import card_wrap, html_escape


def _fmt_money(x: float | None) -> str:
    if x is None or x != x:
        return "N/A"
    return f"{float(x):,.2f}"


def render_macro_reports(
    *,
    reports_dir: Path,
    articles: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    trade_actions: list[dict[str, Any]],
    portfolio_state: Any,
    memory_path: str,
    run_utc: str,
    logger: Any,
) -> None:
    """Write macro_email.html, macro_insights.txt, macro_portfolio_weights.csv, macro_trade_actions.json."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    now_local = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    fx = fetch_usdcad_last()

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("MACRO NEWS + STANDALONE PORTFOLIO (CAD)")
    lines.append("=" * 72)
    lines.append(f"Generated: {now_local}")
    lines.append(f"Memory DB: {memory_path}")
    lines.append("")
    lines.append("HEADLINES (sample)")
    lines.append("-" * 72)
    for a in articles[:12]:
        lines.append(f"- {a.get('title', '')[:120]}")
    lines.append("")
    lines.append("TOP RANKED NAMES")
    lines.append("-" * 72)
    for r in ranked[:15]:
        lines.append(
            f"- {r.get('ticker')} score={r.get('score')} basket={r.get('basket_key')} {r.get('match_reason', '')[:60]}"
        )
    lines.append("")
    lines.append("TARGET WEIGHTS")
    lines.append("-" * 72)
    for t in targets:
        lines.append(f"- {t.get('ticker')} w={float(t.get('weight', 0)):.3f} basket={t.get('basket_key')}")
    lines.append("")
    lines.append("ACTIONS")
    lines.append("-" * 72)
    for ta in trade_actions:
        lines.append(f"- {ta.get('action')} {ta.get('ticker')} sh={ta.get('shares')} @ {ta.get('price_cad')}")
    (reports_dir / "macro_insights.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (reports_dir / "macro_trade_actions.json").write_text(
        json.dumps(trade_actions, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "macro_matches.json").write_text(
        json.dumps({"run_utc": run_utc, "ranked": ranked[:40], "targets": targets}, indent=2),
        encoding="utf-8",
    )

    cash = float(getattr(portfolio_state, "cash_cad", 0.0) or 0.0)
    mkt = 0.0
    rows_csv: list[str] = ["ticker,weight,theme_cluster,basket_key,position_value_cad,macro_theme_key"]
    open_positions = [p for p in portfolio_state.positions if str(p.status) == "OPEN" and float(p.shares) > 0]
    equity = cash
    for p in open_positions:
        px = last_close_cad(p.ticker, fx_usdcad=fx)
        if px is None:
            continue
        v = float(p.shares) * px
        mkt += v
        equity += v
    for p in open_positions:
        px = last_close_cad(p.ticker, fx_usdcad=fx)
        if px is None:
            continue
        v = float(p.shares) * px
        w = (v / equity) if equity > 0 else 0.0
        rows_csv.append(
            f"{p.ticker},{w:.4f},{str(p.macro_theme_cluster or '').replace(',', ';')},"
            f"{str(p.macro_basket_key or '').replace(',', ';')},{v:.2f},{str(p.macro_theme_key or '').replace(',', ';')}"
        )
    (reports_dir / "macro_portfolio_weights.csv").write_text("\n".join(rows_csv) + "\n", encoding="utf-8")

    headline_rows = "".join(
        f"<tr><td style='padding:4px 8px;font-size:12px;color:#374151;'>{html_escape(str(a.get('title',''))[:140])}</td></tr>"
        for a in articles[:10]
    )
    ranked_rows = "".join(
        f"<tr><td style='padding:4px 8px;font-weight:700;'>{html_escape(str(r.get('ticker','')))}</td>"
        f"<td style='padding:4px 8px;font-size:12px;'>{html_escape(str(r.get('basket_key','')))}</td>"
        f"<td style='padding:4px 8px;font-size:12px;'>{r.get('score')}</td></tr>"
        for r in ranked[:12]
    )
    tgt_rows = "".join(
        f"<tr><td style='padding:4px 8px;font-weight:700;'>{html_escape(str(t.get('ticker','')))}</td>"
        f"<td style='padding:4px 8px;'>{float(t.get('weight',0))*100:.1f}%</td>"
        f"<td style='padding:4px 8px;font-size:12px;'>{html_escape(str(t.get('basket_key','')))}</td></tr>"
        for t in targets
    )
    act_rows = "".join(
        f"<tr><td style='padding:4px 8px;color:#059669;font-weight:700;'>{html_escape(str(ta.get('action','')))}</td>"
        f"<td style='padding:4px 8px;font-weight:700;'>{html_escape(str(ta.get('ticker','')))}</td>"
        f"<td style='padding:4px 8px;font-size:12px;'>{html_escape(str(ta.get('reason','')))}</td></tr>"
        for ta in trade_actions[:12]
    )

    inner_head = f"<table style='width:100%;border-collapse:collapse;'>{headline_rows}</table>"
    inner_rank = f"<table style='width:100%;border-collapse:collapse;'>{ranked_rows}</table>"
    inner_tgt = f"<table style='width:100%;border-collapse:collapse;'>{tgt_rows}</table>"
    inner_act = f"<table style='width:100%;border-collapse:collapse;'>{act_rows}</table>" if act_rows else "<div style='font-size:12px;color:#6b7280;'>No trades this run.</div>"

    html = f"""<html>
<body style="font-family:system-ui,-apple-system,Arial,sans-serif;line-height:1.5;color:#111827;
  max-width:900px;margin:0 auto;padding:0;background:#f0f2f5;">
  <div style="background:#111827;padding:18px 24px;border-radius:0 0 14px 14px;">
    <table style="width:100%;border-collapse:collapse;"><tr>
      <td style="vertical-align:middle;padding:0;">
        <span style="font-size:18px;font-weight:800;color:#ffffff;letter-spacing:-0.3px;">Macro Intelligence Report</span>
      </td>
      <td style="vertical-align:middle;text-align:right;padding:0;">
        <span style="font-size:12px;color:#9ca3af;">{html_escape(now_local)}</span>
      </td>
    </tr></table>
  </div>
  <div style="padding:16px 16px 0 16px;">
  {card_wrap("Portfolio snapshot", f"<div style='font-size:13px;color:#374151;'>Cash CAD: <b>{_fmt_money(cash)}</b><br/>"
              f"Estimated equity: <b>{_fmt_money(equity)}</b><br/>Open positions: <b>{len(open_positions)}</b></div>")}
  {card_wrap("Macro headlines", inner_head, accent="#2563eb")}
  {card_wrap("Ranked exposures (hybrid retrieval)", inner_rank, accent="#7c3aed")}
  {card_wrap("Target weights (v1 policy)", inner_tgt, accent="#059669")}
  {card_wrap("Today's macro actions", inner_act, accent="#dc2626")}
  <div style="text-align:center;padding:16px 0 8px 0;font-size:11px;color:#9ca3af;">
    Attachments: macro_insights.txt &bull; macro_portfolio_weights.csv &bull; macro_trade_actions.json
  </div>
  </div>
</body></html>"""
    (reports_dir / "macro_email.html").write_text(html, encoding="utf-8")
    logger.info("Wrote macro reports to %s", reports_dir)

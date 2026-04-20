"""Macro insights + portfolio HTML/text/CSV/JSON reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stock_screener.macro.portfolio_engine import compute_macro_book_metrics
from stock_screener.macro.prices import fetch_usdcad_last, last_close_cad
from stock_screener.reporting.email_style import card_wrap, html_escape


def _fmt_money(x: float | None) -> str:
    if x is None or x != x:
        return "N/A"
    return f"{float(x):,.2f}"


def _dedupe_ranked_by_ticker(ranked: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    """One row per ticker (best score) so the email table is not repeated LQD lines."""
    best: dict[str, dict[str, Any]] = {}
    for r in ranked:
        t = str(r.get("ticker", "")).upper().strip()
        if not t:
            continue
        sc = float(r.get("score", 0) or 0)
        if t not in best or sc > float(best[t].get("score", 0) or 0):
            best[t] = dict(r)
    out = sorted(best.values(), key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    return out[:limit]


def _collapse_ws(s: str) -> str:
    return " ".join(str(s).split())


def _trim_visible(s: str, max_len: int) -> str:
    t = _collapse_ws(s)
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _decision_narrative_for_email(d: dict[str, Any]) -> str:
    """PM REASON is often empty; fall back to bull/bear/risk text the models already produced."""
    r = str(d.get("reasoning", "")).strip()
    if r:
        return _trim_visible(r, 900)
    parts: list[str] = []
    bull = str(d.get("bull_thesis", "")).strip()
    bear = str(d.get("bear_thesis", "")).strip()
    risk = str(d.get("risk_assessment", "")).strip()
    facets = str(d.get("analyst_facets", "")).strip()
    if facets:
        parts.append("Analyst facets: " + _trim_visible(facets, 400))
    if bull:
        parts.append("Bull: " + _trim_visible(bull, 420))
    if bear:
        parts.append("Bear: " + _trim_visible(bear, 420))
    if risk:
        parts.append("Risk: " + _trim_visible(risk, 420))
    if parts:
        return "\n".join(parts)
    return "No PM or debate narrative was returned for this name."


def _portfolio_reasoning_html(pr: Any) -> str:
    if not isinstance(pr, dict) or not pr:
        return ""
    blocks: list[str] = []
    labels = [
        ("overall", "Overall"),
        ("adjustments", "Adjustments"),
        ("concentration_risk", "Concentration risk"),
        ("correlation_flag", "Correlation"),
        ("regime_check", "Regime"),
    ]
    for key, title in labels:
        v = pr.get(key)
        if v and str(v).strip():
            blocks.append(
                f"<p style='margin:8px 0;font-size:12px;color:#374151;line-height:1.45;'>"
                f"<b>{html_escape(title)}</b><br/>{html_escape(_trim_visible(str(v), 720))}</p>"
            )
    tw = pr.get("ticker_weights")
    if isinstance(tw, dict) and tw:
        blocks.append(
            f"<p style='margin:8px 0;font-size:12px;color:#374151;'><b>Suggested weights</b><br/>"
            f"{html_escape(_trim_visible(str(tw), 500))}</p>"
        )
    ex = pr.get("excluded")
    if isinstance(ex, list) and ex:
        blocks.append(
            f"<p style='margin:8px 0;font-size:12px;color:#374151;'><b>Excluded</b> "
            f"{html_escape(', '.join(str(x) for x in ex[:16]))}</p>"
        )
    raw = pr.get("raw_response")
    if not blocks and raw and str(raw).strip():
        blocks.append(
            f"<p style='margin:8px 0;font-size:12px;color:#374151;'><b>Portfolio model (excerpt)</b><br/>"
            f"{html_escape(_trim_visible(str(raw), 900))}</p>"
        )
    return "".join(blocks)


def _target_weights_card_title(llm_agent: dict[str, Any] | None) -> str:
    if not isinstance(llm_agent, dict) or llm_agent.get("status") in (None, "disabled"):
        return "Target weights (rule book)"
    st = str(llm_agent.get("status", ""))
    primary = bool(llm_agent.get("primary_mode"))
    if st == "skipped":
        return "Target weights (rule book; LLM skipped)"
    if st == "fallback":
        reason = str(llm_agent.get("reason", ""))
        if "no_buy_or_overweight" in reason:
            return "Target weights (rule book; no LLM BUY/OVERWEIGHT)"
        return "Target weights (rule book; LLM fallback)"
    if st == "success" and primary:
        return "Target weights (LLM-primary)"
    if st == "success":
        return "Target weights (LLM score blend + rule allocator)"
    return "Target weights"


def _fmt_signed_cad(x: float) -> str:
    return f"{float(x):+,.2f}"


def _macro_performance_txt_lines(book: dict[str, Any], *, n_hist: int) -> list[str]:
    lines = [
        "PAPER PORTFOLIO PERFORMANCE (vs initial budget)",
        "-" * 72,
        f"Initial budget CAD: {_fmt_money(book['initial_budget_cad'])}",
        f"Equity (mark-to-market) CAD: {_fmt_money(book['equity_cad'])}",
        f"Net P/L CAD: {_fmt_signed_cad(float(book['net_pl_cad']))}  ({float(book['total_return_pct']):+.2f}%)",
        f"Unrealized on open names CAD: {_fmt_signed_cad(float(book['unrealized_pl_cad']))}",
        f"Residual (realized + cash drag) CAD: {_fmt_signed_cad(float(book['realized_pl_cad']))}",
        f"Cash CAD: {_fmt_money(book['cash_cad'])}  |  Open positions MTM CAD: {_fmt_money(book['open_market_value_cad'])}",
        f"Open cost basis CAD: {_fmt_money(book['open_cost_basis_cad'])}",
        f"PnL snapshots stored: {n_hist} (one per macro run when not dry-run)",
    ]
    de = book.get("delta_vs_prior_equity_cad")
    if de is not None:
        lines.append(f"Δ equity vs prior snapshot CAD: {_fmt_signed_cad(float(de))}")
    rows = book.get("position_rows") or []
    if rows:
        lines.append("")
        lines.append("OPEN POSITIONS (MTM)")
        lines.append("-" * 72)
        for r in rows:
            lines.append(
                f"- {r.get('ticker')}  value={_fmt_money(float(r.get('value_cad', 0)))}  "
                f"uPnL={_fmt_signed_cad(float(r.get('unrealized_pl_cad', 0)))}  "
                f"({float(r.get('unrealized_pct', 0)):+.1f}%)  days={r.get('days_held')}"
            )
    return lines


def _macro_performance_html(book: dict[str, Any], *, n_hist: int) -> str:
    de = book.get("delta_vs_prior_equity_cad")
    delta_html = ""
    if de is not None:
        c = "#059669" if float(de) >= 0 else "#dc2626"
        delta_html = (
            f"<div style='font-size:13px;margin:6px 0;'>Δ vs prior snapshot: "
            f"<b style='color:{c};'>{html_escape(_fmt_signed_cad(float(de)))}</b> CAD</div>"
        )
    pos_rows = book.get("position_rows") or []
    pos_table = ""
    if pos_rows:
        parts = [
            "<tr style='background:#f9fafb;'>"
            "<th style='text-align:left;padding:6px 8px;font-size:11px;'>Ticker</th>"
            "<th style='text-align:right;padding:6px 8px;font-size:11px;'>Value</th>"
            "<th style='text-align:right;padding:6px 8px;font-size:11px;'>uPnL</th>"
            "<th style='text-align:right;padding:6px 8px;font-size:11px;'>%</th>"
            "<th style='text-align:right;padding:6px 8px;font-size:11px;'>Days</th></tr>"
        ]
        for r in pos_rows:
            up = float(r.get("unrealized_pl_cad", 0))
            col = "#059669" if up >= 0 else "#dc2626"
            parts.append(
                f"<tr><td style='padding:6px 8px;font-weight:700;'>{html_escape(str(r.get('ticker','')))}</td>"
                f"<td style='padding:6px 8px;text-align:right;font-size:12px;'>{html_escape(_fmt_money(float(r.get('value_cad',0))))}</td>"
                f"<td style='padding:6px 8px;text-align:right;font-size:12px;color:{col};'>"
                f"{html_escape(_fmt_signed_cad(up))}</td>"
                f"<td style='padding:6px 8px;text-align:right;font-size:12px;color:{col};'>"
                f"{float(r.get('unrealized_pct', 0)):+.1f}%</td>"
                f"<td style='padding:6px 8px;text-align:right;font-size:12px;'>{html_escape(str(r.get('days_held','')))}</td></tr>"
            )
        pos_table = f"<table style='width:100%;border-collapse:collapse;margin-top:10px;'>{''.join(parts)}</table>"
    net = float(book["net_pl_cad"])
    nc = "#059669" if net >= 0 else "#dc2626"
    unr = float(book["unrealized_pl_cad"])
    urc = "#059669" if unr >= 0 else "#dc2626"
    return f"""<div style='font-size:13px;color:#374151;line-height:1.55;'>
  <div>Initial budget: <b>{html_escape(_fmt_money(float(book['initial_budget_cad'])))}</b> CAD</div>
  <div>Equity (MTM): <b>{html_escape(_fmt_money(float(book['equity_cad'])))}</b> CAD</div>
  <div>Net P/L: <b style='color:{nc};'>{html_escape(_fmt_signed_cad(net))}</b> CAD
    &nbsp;(<b style='color:{nc};'>{float(book['total_return_pct']):+.2f}%</b> vs budget)</div>
  <div>Unrealized (open): <b style='color:{urc};'>{html_escape(_fmt_signed_cad(unr))}</b> CAD</div>
  <div style='font-size:12px;color:#6b7280;'>Cash {_fmt_money(float(book['cash_cad']))} · Open MTM {_fmt_money(float(book['open_market_value_cad']))} · Cost basis {_fmt_money(float(book['open_cost_basis_cad']))}</div>
  <div style='font-size:12px;color:#6b7280;'>Stored PnL snapshots: {n_hist}</div>
  {delta_html}
  {pos_table}
  </div>"""


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
    llm_agent: dict[str, Any] | None = None,
    initial_cash_cad: float = 500.0,
) -> None:
    """Write macro_email.html, macro_insights.txt, macro_portfolio_weights.csv, macro_trade_actions.json."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    now_local = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    fx = fetch_usdcad_last()
    hist_pre = list(getattr(portfolio_state, "pnl_history", None) or [])
    prior_eq: float | None = None
    if len(hist_pre) >= 2:
        try:
            prior_eq = float(hist_pre[-2].get("equity_cad", 0) or 0)
        except (TypeError, ValueError):
            prior_eq = None
    book = compute_macro_book_metrics(
        portfolio_state,
        fx=fx,
        initial_cash_cad=float(initial_cash_cad),
        prior_equity_for_delta=prior_eq,
    )

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("MACRO NEWS + STANDALONE PORTFOLIO (CAD)")
    lines.append("=" * 72)
    lines.append(f"Generated: {now_local}")
    lines.append(f"Memory DB: {memory_path}")
    lines.append("")
    lines.extend(_macro_performance_txt_lines(book, n_hist=len(hist_pre)))
    lines.append("")
    lines.append("HEADLINES (sample)")
    lines.append("-" * 72)
    for a in articles[:12]:
        lines.append(f"- {a.get('title', '')[:120]}")
    lines.append("")
    lines.append("TOP RANKED NAMES")
    lines.append("-" * 72)
    for r in _dedupe_ranked_by_ticker(ranked, limit=15):
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
    if isinstance(llm_agent, dict) and llm_agent.get("status") not in (None, "disabled"):
        lines.append("")
        lines.append("LLM AGENT (same stack as daily screener)")
        lines.append("-" * 72)
        lines.append(f"status={llm_agent.get('status')} primary_mode={llm_agent.get('primary_mode')}")
        dec = llm_agent.get("decisions") or {}
        if isinstance(dec, dict):
            for tk, d in list(dec.items())[:12]:
                if isinstance(d, dict):
                    lines.append(f"- {tk}: {d.get('rating')} score={d.get('score')}")
                    nar = _decision_narrative_for_email(d)
                    for ln in nar.split("\n"):
                        lines.append(f"    {ln[:500]}")
        pr = llm_agent.get("portfolio_reasoning")
        if isinstance(pr, dict) and pr:
            lines.append("")
            lines.append("PORTFOLIO-LEVEL LLM")
            lines.append("-" * 72)
            for key in ("overall", "adjustments", "concentration_risk", "correlation_flag", "regime_check"):
                v = pr.get(key)
                if v and str(v).strip():
                    lines.append(f"- {key}: {str(v)[:400]}")
    (reports_dir / "macro_insights.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (reports_dir / "macro_trade_actions.json").write_text(
        json.dumps(trade_actions, indent=2),
        encoding="utf-8",
    )
    _matches: dict[str, Any] = {
        "run_utc": run_utc,
        "ranked": ranked[:40],
        "targets": targets,
        "performance": book,
        "pnl_history_tail": hist_pre[-14:] if len(hist_pre) > 14 else list(hist_pre),
    }
    if isinstance(llm_agent, dict):
        _matches["llm_agent"] = llm_agent
    (reports_dir / "macro_matches.json").write_text(json.dumps(_matches, indent=2), encoding="utf-8")

    equity = float(book["equity_cad"])
    cash = float(book["cash_cad"])
    n_open = int(book["n_open"])
    rows_csv: list[str] = [
        "ticker,weight,theme_cluster,basket_key,position_value_cad,macro_theme_key,unrealized_pl_cad,pct_vs_entry"
    ]
    for r in book.get("position_rows") or []:
        v = float(r.get("value_cad", 0))
        w = (v / equity) if equity > 0 else 0.0
        rows_csv.append(
            f"{r.get('ticker')},{w:.4f},{str(r.get('macro_theme_cluster', '') or '').replace(',', ';')},"
            f"{str(r.get('macro_basket_key', '') or '').replace(',', ';')},{v:.2f},"
            f"{str(r.get('macro_theme_key', '') or '').replace(',', ';')},"
            f"{float(r.get('unrealized_pl_cad', 0)):.2f},{float(r.get('unrealized_pct', 0)):.2f}"
        )
    (reports_dir / "macro_portfolio_weights.csv").write_text("\n".join(rows_csv) + "\n", encoding="utf-8")

    headline_rows = "".join(
        f"<tr><td style='padding:4px 8px;font-size:12px;color:#374151;'>{html_escape(str(a.get('title',''))[:140])}</td></tr>"
        for a in articles[:10]
    )
    ranked_display = _dedupe_ranked_by_ticker(ranked, limit=12)
    ranked_rows = "".join(
        f"<tr><td style='padding:4px 8px;font-weight:700;'>{html_escape(str(r.get('ticker','')))}</td>"
        f"<td style='padding:4px 8px;font-size:12px;'>{html_escape(str(r.get('basket_key','')))}</td>"
        f"<td style='padding:4px 8px;font-size:12px;'>{r.get('score')}</td></tr>"
        for r in ranked_display
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

    llm_block = ""
    if isinstance(llm_agent, dict) and llm_agent.get("status") not in (None, "disabled"):
        dec = llm_agent.get("decisions") or {}
        ticker_blocks: list[str] = []
        if isinstance(dec, dict):
            for tk, d in list(dec.items())[:10]:
                if isinstance(d, dict):
                    nar = html_escape(_decision_narrative_for_email(d))
                    pm_m = html_escape(str(d.get("pm_model", "") or ""))
                    pm_bit = f"<span style='color:#6b7280;font-size:11px;'>{pm_m}</span>" if pm_m else ""
                    ticker_blocks.append(
                        f"<div style='margin:0 0 14px 0;padding-bottom:12px;border-bottom:1px solid #e5e7eb;'>"
                        f"<div style='font-size:13px;margin-bottom:6px;'>"
                        f"<b>{html_escape(str(tk))}</b> "
                        f"<span style='color:#92400e;font-weight:600;'>{html_escape(str(d.get('rating','')))}</span>"
                        f" &nbsp;score {html_escape(str(d.get('score','')))} {pm_bit}</div>"
                        f"<div style='font-size:12px;color:#374151;line-height:1.5;white-space:pre-wrap;'>{nar}</div>"
                        f"</div>"
                    )
        pr_html = _portfolio_reasoning_html(llm_agent.get("portfolio_reasoning"))
        pr_section = ""
        if pr_html:
            pr_section = (
                f"<div style='margin-top:14px;padding-top:12px;border-top:1px solid #e5e7eb;'>"
                f"<div style='font-size:13px;font-weight:700;margin-bottom:6px;color:#111827;'>Portfolio-level reasoning</div>"
                f"{pr_html}</div>"
            )
        _llm_empty = "<div style='padding:8px;font-size:12px;color:#6b7280;'>No per-ticker decisions recorded.</div>"
        inner_llm = (
            f"<p style='font-size:12px;color:#374151;margin:0 0 12px 0;'>status={html_escape(str(llm_agent.get('status')))} "
            f"&nbsp; primary_mode={html_escape(str(llm_agent.get('primary_mode')))}</p>"
            f"{''.join(ticker_blocks) or _llm_empty}"
            f"{pr_section}"
        )
        llm_block = card_wrap("LLM portfolio layer (Groq / Gemini)", inner_llm, accent="#b45309")

    inner_perf = _macro_performance_html(book, n_hist=len(hist_pre))

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
              f"Estimated equity: <b>{_fmt_money(equity)}</b><br/>Open positions: <b>{n_open}</b></div>")}
  {card_wrap("Paper portfolio performance (vs initial budget)", inner_perf, accent="#0f766e")}
  {card_wrap("Macro headlines", inner_head, accent="#2563eb")}
  {card_wrap("Ranked exposures (hybrid retrieval, best row per name)", inner_rank, accent="#7c3aed")}
  {card_wrap(html_escape(_target_weights_card_title(llm_agent)), inner_tgt, accent="#059669")}
  {llm_block}
  {card_wrap("Today's macro actions", inner_act, accent="#dc2626")}
  <div style="text-align:center;padding:16px 0 8px 0;font-size:11px;color:#9ca3af;">
    Attachments: macro_insights.txt &bull; macro_portfolio_weights.csv &bull; macro_trade_actions.json
  </div>
  </div>
</body></html>"""
    (reports_dir / "macro_email.html").write_text(html, encoding="utf-8")
    logger.info("Wrote macro reports to %s", reports_dir)

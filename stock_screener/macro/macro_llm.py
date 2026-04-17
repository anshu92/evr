"""Bridge macro retrieval + headlines into the same LLM trading agent as daily.py."""

from __future__ import annotations

import math
import os
from typing import Any

import pandas as pd

from stock_screener.macro.config import MacroConfig
from stock_screener.macro.portfolio_engine import build_target_weights
from stock_screener.macro.prices import fetch_usdcad_last, last_close_cad
from stock_screener.macro.retrieve import load_universe
from stock_screener.portfolio.state import load_portfolio_state


def _macro_digest(articles: list[dict[str, Any]], *, max_items: int = 40, max_chars: int = 6000) -> str:
    lines: list[str] = []
    n = 0
    for a in articles[:max_items]:
        title = str(a.get("title", "")).strip()
        if not title:
            continue
        src = str(a.get("publisher") or a.get("source_name", ""))
        lines.append(f"- [{src}] {title}")
        n += 1
        if sum(len(x) for x in lines) >= max_chars:
            break
    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[: max_chars - 3] + "..."
    return body or "(no headlines this run)"


def _best_ranked_row_per_ticker(ranked: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for r in ranked:
        t = str(r.get("ticker", "")).upper()
        if not t:
            continue
        sc = float(r.get("score", 0) or 0)
        if t not in best or sc > float(best[t].get("score", 0) or 0):
            best[t] = dict(r)
    return best


def _yf_quick_features(ticker: str, fx: float | None) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        import yfinance as yf

        h = yf.Ticker(ticker).history(period="120d", auto_adjust=True)
        if h is None or h.empty or len(h) < 5:
            return out
        close = h["Close"].astype(float)
        last = float(close.iloc[-1])
        px_cad = last_close_cad(ticker, fx_usdcad=fx)
        if px_cad is None or not math.isfinite(px_cad):
            px_cad = last
        out["last_close_cad"] = float(px_cad)
        r5 = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
        r20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else r5
        out["ret_5d"] = r5
        out["ret_10d"] = float(close.iloc[-1] / close.iloc[-11] - 1.0) if len(close) >= 11 else r5
        out["ret_20d"] = r20
        out["ret_60d"] = float(close.iloc[-1] / close.iloc[-61] - 1.0) if len(close) >= 61 else r20
        out["ret_120d"] = out["ret_60d"]
        vol = close.pct_change().dropna()
        if len(vol) >= 10:
            out["vol_20d_ann"] = min(2.0, float(vol.iloc[-20:].std() * (252.0**0.5)))
            out["vol_60d_ann"] = min(2.0, float(vol.iloc[-60:].std() * (252.0**0.5)))
        else:
            out["vol_20d_ann"] = 0.25
            out["vol_60d_ann"] = 0.25
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_g = float(gain.rolling(14).mean().iloc[-1] or 0)
        avg_l = float(loss.rolling(14).mean().iloc[-1] or 1e-9)
        rs = avg_g / max(avg_l, 1e-9)
        out["rsi_14"] = float(100.0 - (100.0 / (1.0 + rs)))
        out["ma20_ratio"] = float(close.iloc[-1] / close.iloc[-20:].mean() - 1.0) if len(close) >= 20 else 0.0
        out["ma50_ratio"] = float(close.iloc[-1] / close.iloc[-50:].mean() - 1.0) if len(close) >= 50 else 0.0
        out["ma200_ratio"] = float(close.iloc[-1] / close.iloc[-60:].mean() - 1.0) if len(close) >= 60 else 0.0
        roll_max = close.rolling(60, min_periods=5).max().iloc[-1]
        roll_min = close.rolling(60, min_periods=5).min().iloc[-1]
        out["drawdown_60d"] = float(close.iloc[-1] / roll_max - 1.0) if roll_max and roll_max > 0 else 0.0
        out["dist_52w_high"] = float(close.iloc[-1] / roll_max - 1.0) if roll_max and roll_max > 0 else 0.0
        out["dist_52w_low"] = float(close.iloc[-1] / roll_min - 1.0) if roll_min and roll_min > 0 else 0.0
    except Exception:
        pass
    return out


def build_macro_agent_candidates(
    *,
    ranked: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    universe: dict[str, dict[str, Any]],
    primary_theme_key: str,
    max_tickers: int,
    fx_usdcad: float | None,
) -> list[dict[str, Any]]:
    """Build per-ticker feature dicts for ``analyze_candidates`` (same shape as daily ML rows, macro-filled)."""
    digest = _macro_digest(articles)
    best = _best_ranked_row_per_ticker(ranked)
    tickers = sorted(best.keys(), key=lambda t: float(best[t].get("score", 0) or 0), reverse=True)[:max_tickers]

    candidates: list[dict[str, Any]] = []
    for t in tickers:
        row = best.get(t, {})
        sc = float(row.get("score", 0) or 0)
        meta = universe.get(t, {})
        sector = str(meta.get("sector", "Unknown"))
        industry = str(meta.get("industry", "Unknown"))
        yf_f = _yf_quick_features(t, fx_usdcad)
        px = yf_f.get("last_close_cad")
        if px is None or not math.isfinite(px):
            px = last_close_cad(t, fx_usdcad=fx_usdcad) or 0.0
        pred_ret = max(-0.08, min(0.08, (sc - 0.72) * 0.02))
        pred_conf = max(0.35, min(0.92, 0.4 + sc * 0.45))
        mr = str(row.get("match_reason", ""))[:220]
        bk = str(row.get("basket_key", ""))
        tc = str(row.get("theme_cluster", ""))
        news_headlines: list[dict[str, Any]] = [
            {
                "title": f"[MACRO RUN] Theme={primary_theme_key!s} | Basket={bk} | Cluster={tc}",
                "publisher": "macro_insights",
                "source_type": "macro",
                "publish_date": "",
            },
            {"title": digest[:2800], "publisher": "macro_headlines", "source_type": "macro", "publish_date": ""},
            {
                "title": f"[{t}] Hybrid retrieval: {mr}",
                "publisher": "macro_retrieval",
                "source_type": "macro",
                "publish_date": "",
            },
        ]
        c: dict[str, Any] = {
            "ticker": t,
            "pred_return": pred_ret,
            "pred_confidence": pred_conf,
            "pred_peak_days": 3.0,
            "score": sc,
            "last_close_cad": float(px or 0.0),
            "ret_60d": yf_f.get("ret_60d", 0.0),
            "ret_5d": yf_f.get("ret_5d", 0.0),
            "ret_10d": yf_f.get("ret_10d", 0.0),
            "ret_20d": yf_f.get("ret_20d", 0.0),
            "ret_120d": yf_f.get("ret_120d", 0.0),
            "vol_20d_ann": yf_f.get("vol_20d_ann", float("nan")),
            "vol_60d_ann": yf_f.get("vol_60d_ann", float("nan")),
            "rsi_14": yf_f.get("rsi_14", 50.0),
            "beta": float(meta.get("beta", 1.0) or 1.0) if meta.get("beta") is not None else 1.0,
            "log_market_cap": float(meta.get("log_market_cap", float("nan")))
            if meta.get("log_market_cap") is not None
            else float("nan"),
            "ma20_ratio": yf_f.get("ma20_ratio", 0.0),
            "ma50_ratio": yf_f.get("ma50_ratio", 0.0),
            "ma200_ratio": yf_f.get("ma200_ratio", 0.0),
            "drawdown_60d": yf_f.get("drawdown_60d", 0.0),
            "dist_52w_high": yf_f.get("dist_52w_high", 0.0),
            "dist_52w_low": yf_f.get("dist_52w_low", 0.0),
            "market_vol_regime": 1.0,
            "market_trend_20d": 0.0,
            "market_breadth": 0.5,
            "news_sentiment_avg": float("nan"),
            "news_volume_5d": 0.0,
            "insider_net_buys_90d": float("nan"),
            "insider_buy_ratio_90d": float("nan"),
            "insider_activity_recency": float("nan"),
            "trailing_pe": float("nan"),
            "forward_pe": float("nan"),
            "price_to_book": float("nan"),
            "profit_margins": float("nan"),
            "return_on_equity": float("nan"),
            "debt_to_equity": float("nan"),
            "revenue_growth": float("nan"),
            "earnings_growth": float("nan"),
            "dividend_yield": float("nan"),
            "recommendation_mean": float("nan"),
            "num_analyst_opinions": float("nan"),
            "sector": sector,
            "industry": industry,
            "news_headlines": news_headlines,
        }
        candidates.append(c)
    return candidates


def _ranked_to_screened(ranked: list[dict[str, Any]], tickers: set[str]) -> pd.DataFrame:
    best = _best_ranked_row_per_ticker(ranked)
    rows: list[dict[str, Any]] = []
    idx: list[str] = []
    for t in tickers:
        u = t.upper()
        r = best.get(u, {})
        rows.append(
            {
                "score": float(r.get("score", 0) or 0),
                "pred_return": max(-0.1, min(0.1, (float(r.get("score", 0) or 0) - 0.72) * 0.02)),
                "pred_confidence": max(0.35, min(0.92, 0.4 + float(r.get("score", 0) or 0) * 0.45)),
                "vol_20d_ann": 0.2,
                "sector": str(r.get("theme_cluster") or "macro"),
            }
        )
        idx.append(u)
    return pd.DataFrame(rows, index=idx)


def _clip_meta_text(s: str, n: int) -> str:
    t = str(s or "").strip()
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


def _decisions_to_meta(decisions: dict[str, Any]) -> dict[str, Any]:
    from stock_screener.agents.trading_agent import AgentDecision

    out: dict[str, dict[str, Any]] = {}
    for t, d in decisions.items():
        if isinstance(d, AgentDecision):
            row: dict[str, Any] = {
                "rating": d.rating,
                "score": d.score,
                "reasoning": d.reasoning,
                "position_size": getattr(d, "position_size", "MEDIUM"),
                "target_weight": getattr(d, "target_weight", None),
                "pm_model": getattr(d, "pm_model", ""),
                "bull_thesis": _clip_meta_text(d.bull_thesis, 450),
                "bear_thesis": _clip_meta_text(d.bear_thesis, 450),
                "risk_assessment": _clip_meta_text(d.risk_assessment, 450),
            }
            ar = getattr(d, "analyst_reports", None)
            if isinstance(ar, dict) and ar:
                facets = [f"{k}: {_clip_meta_text(str(v), 100)}" for k, v in ar.items() if str(v).strip()]
                if facets:
                    row["analyst_facets"] = " | ".join(facets[:6])
            out[t] = row
    return out


def normalize_macro_target_list(targets: list[dict[str, Any]], cfg: MacroConfig) -> list[dict[str, Any]]:
    """Apply the same gross / per-name caps as ``build_target_weights`` tail."""
    if not targets:
        return []
    gross = min(cfg.target_gross_exposure, 1.0 - cfg.min_cash_weight)
    for p in targets:
        cap = cfg.etf_weight_cap if p.get("asset_type") == "etf" else cfg.stock_weight_cap
        p["weight"] = min(float(p.get("weight", 0) or 0), cap)
    etf_weight = sum(float(p["weight"]) for p in targets if p.get("asset_type") == "etf")
    if etf_weight < cfg.min_etf_weight * gross and targets:
        scale = (cfg.min_etf_weight * gross) / max(etf_weight, 1e-9)
        scale = min(scale, 1.5)
        for p in targets:
            if p.get("asset_type") == "etf":
                p["weight"] = min(cfg.etf_weight_cap, float(p["weight"]) * scale)
    total = sum(float(p["weight"]) for p in targets)
    if total > gross:
        f = gross / total
        for p in targets:
            p["weight"] = float(p["weight"]) * f
    return targets


def _targets_from_llm_weights(
    tw: pd.DataFrame,
    best_by_ticker: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for t in tw.index:
        u = str(t).upper()
        meta = best_by_ticker.get(u, {})
        w = float(tw.loc[t, "weight"])
        targets.append(
            {
                "ticker": u,
                "weight": w,
                "basket_key": str(meta.get("basket_key", "")),
                "theme_cluster": str(meta.get("theme_cluster", "")),
                "score": float(meta.get("score", 0) or 0),
                "asset_type": str(meta.get("asset_type", "etf")),
            }
        )
    return targets


def refine_macro_targets_with_llm(
    *,
    cfg: MacroConfig,
    all_ranked: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    primary_theme_key: str,
    rule_targets: list[dict[str, Any]],
    log: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Optionally run the same Groq/Gemini agent stack as daily; returns (targets, llm_meta, ranked_out)."""
    import logging

    _log = log or logging.getLogger(__name__)
    if not cfg.llm_agent_enabled:
        return rule_targets, {"status": "disabled"}, all_ranked

    from stock_screener.agents.config import get_agent_config
    from stock_screener.agents.trading_agent import (
        analyze_candidates,
        analyze_portfolio,
        blend_llm_scores,
        build_portfolio_context,
        compute_llm_primary_weights,
        get_model_usage,
        select_tickers_llm_primary,
    )

    agent_cfg = get_agent_config()
    if not agent_cfg.get("api_key"):
        _log.warning("Macro LLM: no agent API key (GROQ/OPENROUTER/GEMINI); using rule targets")
        return rule_targets, {"status": "skipped", "reason": "no API key configured"}, all_ranked

    fx = fetch_usdcad_last()
    universe = load_universe()
    candidates = build_macro_agent_candidates(
        ranked=all_ranked,
        articles=articles,
        universe=universe,
        primary_theme_key=primary_theme_key,
        max_tickers=cfg.llm_max_tickers,
        fx_usdcad=fx,
    )
    if not candidates:
        return rule_targets, {"status": "skipped", "reason": "no candidates"}, all_ranked

    state = load_portfolio_state(cfg.portfolio_state_path, initial_cash_cad=cfg.portfolio_budget_cad)
    portfolio_ctx = build_portfolio_context(state.positions, state.cash_cad)

    primary_mode = cfg.llm_decision_primary
    decisions = analyze_candidates(
        candidates,
        portfolio_context=portfolio_ctx,
        log=_log,
        primary_mode=primary_mode,
        config=agent_cfg,
    )
    if not decisions:
        return rule_targets, {"status": "no_results", "reason": "LLM returned no decisions"}, all_ranked

    tickers_df = set(str(c["ticker"]).upper() for c in candidates) | {str(t["ticker"]).upper() for t in rule_targets}
    screened = _ranked_to_screened(all_ranked, tickers_df)
    for c in candidates:
        t = str(c["ticker"]).upper()
        if t not in screened.index:
            continue
        screened.at[t, "score"] = float(c.get("score", screened.at[t, "score"]))

    llm_meta: dict[str, Any] = {
        "status": "success",
        "n_analyzed": len(decisions),
        "primary_mode": primary_mode,
        "decisions": _decisions_to_meta(decisions),
    }
    try:
        llm_meta["model_usage"] = get_model_usage()
    except Exception:
        pass

    best_by = _best_ranked_row_per_ticker(all_ranked)

    if primary_mode:
        has_conviction = any(
            getattr(d, "rating", "") in ("BUY", "OVERWEIGHT") for d in decisions.values()
        )
        if not has_conviction:
            _log.warning(
                "Macro LLM-primary: no BUY/OVERWEIGHT in analyzed set; using rule targets "
                "(LLM decisions retained in llm_agent meta). Set MACRO_LLM_PRIMARY=0 for "
                "macro-only blend, or LLM_DECISION_PRIMARY=0 when MACRO_LLM_PRIMARY is unset."
            )
            llm_meta["status"] = "fallback"
            llm_meta["reason"] = "no_buy_or_overweight_in_llm_primary"
            return rule_targets, llm_meta, all_ranked

        selected = select_tickers_llm_primary(
            screened, decisions, max_positions=cfg.max_positions, log=_log
        )
        if not selected:
            _log.warning("Macro LLM-primary: no tickers selected; using rule targets")
            llm_meta["status"] = "fallback"
            llm_meta["reason"] = "no BUY/HOLD selection"
            return rule_targets, llm_meta, all_ranked

        tw = compute_llm_primary_weights(
            selected,
            decisions,
            screened,
            max_position_pct=max(cfg.etf_weight_cap, cfg.stock_weight_cap, 0.20),
            min_position_pct=0.02,
            log=_log,
        )
        if tw is None or tw.empty:
            return rule_targets, {**llm_meta, "status": "fallback", "reason": "empty weights"}, all_ranked

        if cfg.llm_portfolio_reasoning and bool(agent_cfg.get("portfolio_reasoning", True)):
            try:
                mcond = {
                    "vol_regime": 1.0,
                    "market_trend": 0.0,
                    "breadth": 0.5,
                }
                pr = analyze_portfolio(
                    candidates,
                    decisions,
                    portfolio_ctx,
                    market_conditions=mcond,
                    max_positions=cfg.max_positions,
                    budget_cad=cfg.portfolio_budget_cad,
                    log=_log,
                    primary_mode=True,
                    config=agent_cfg,
                )
                if pr:
                    llm_meta["portfolio_reasoning"] = pr
                    if cfg.llm_portfolio_enforce:
                        from stock_screener.agents.trading_agent import apply_portfolio_reasoning_enforced

                        tw = apply_portfolio_reasoning_enforced(
                            tw,
                            pr,
                            screened,
                            decisions=decisions,
                            sector_cap=0.35,
                            regime_reduce_scalar=0.5,
                            max_position_pct=max(cfg.etf_weight_cap, cfg.stock_weight_cap, 0.20),
                            log=_log,
                        )
            except Exception as e:
                _log.debug("Macro portfolio LLM skipped: %s", e)

        targets = normalize_macro_target_list(_targets_from_llm_weights(tw, best_by), cfg)
        _log.info("Macro LLM-primary targets: %s", [t["ticker"] for t in targets])
        return targets, llm_meta, all_ranked

    screened2 = blend_llm_scores(
        screened.copy(),
        decisions,
        score_col="score",
        ml_weight=cfg.llm_ml_weight,
        llm_weight=cfg.llm_llm_weight,
        log=_log,
    )
    score_map = {str(i).upper(): float(screened2.loc[i, "score"]) for i in screened2.index}
    ranked_out = []
    for r in all_ranked:
        rr = dict(r)
        u = str(rr.get("ticker", "")).upper()
        if u in score_map:
            rr["score"] = score_map[u]
        ranked_out.append(rr)
    ranked_out.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    targets = build_target_weights(ranked=ranked_out, cfg=cfg)
    _log.info("Macro LLM-blend targets: %s", [t["ticker"] for t in targets])
    return targets, llm_meta, ranked_out

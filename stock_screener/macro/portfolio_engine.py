"""Macro target weights and paper trade execution against isolated portfolio state."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from stock_screener.macro.config import MacroConfig
from stock_screener.macro.prices import fetch_usdcad_last, last_close_cad
from stock_screener.portfolio.state import (
    Position,
    append_portfolio_events,
    load_portfolio_state,
    resolve_portfolio_event_log_path,
    save_portfolio_state,
)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _open_positions(state: Any) -> list[Position]:
    return [p for p in state.positions if str(p.status) == "OPEN" and float(p.shares) > 0]


def _equity_cad(state: Any, fx: float) -> float:
    cash = float(state.cash_cad)
    mkt = 0.0
    for p in _open_positions(state):
        px = last_close_cad(p.ticker, fx_usdcad=fx)
        if px is None:
            continue
        mkt += float(p.shares) * px
    return cash + mkt


def build_target_weights(
    *,
    ranked: list[dict[str, Any]],
    cfg: MacroConfig,
    max_themes: int = 4,
) -> list[dict[str, Any]]:
    """Pick ETF-first targets with score gaps; returns list of {ticker, weight, basket_key, theme_cluster}."""
    if not ranked:
        return []

    etfs = [r for r in ranked if r.get("asset_type") == "etf"]
    stocks = [r for r in ranked if r.get("asset_type") != "etf"]

    picks: list[dict[str, Any]] = []
    used_baskets: set[str] = set()

    for r in etfs:
        bk = str(r.get("basket_key", ""))
        if bk in used_baskets:
            continue
        used_baskets.add(bk)
        picks.append(
            {
                "ticker": str(r["ticker"]).upper(),
                "weight": 0.0,
                "basket_key": bk,
                "theme_cluster": str(r.get("theme_cluster", "")),
                "score": float(r.get("score", 0.0)),
                "asset_type": "etf",
            }
        )
        if len(picks) >= max_themes:
            break

    n_stock_slots = min(cfg.single_stock_max_positions, max(0, cfg.max_positions - len(picks)))
    for r in stocks:
        if n_stock_slots <= 0:
            break
        bk = str(r.get("basket_key", ""))
        if bk in used_baskets:
            continue
        if float(r.get("score", 0)) < 0.55:
            continue
        used_baskets.add(bk)
        picks.append(
            {
                "ticker": str(r["ticker"]).upper(),
                "weight": 0.0,
                "basket_key": bk,
                "theme_cluster": str(r.get("theme_cluster", "")),
                "score": float(r.get("score", 0.0)),
                "asset_type": "stock",
            }
        )
        n_stock_slots -= 1
        if len(picks) >= cfg.max_positions:
            break

    if not picks:
        return []

    gross = min(cfg.target_gross_exposure, 1.0 - cfg.min_cash_weight)
    n = len(picks)
    base = gross / n
    for p in picks:
        cap = cfg.etf_weight_cap if p["asset_type"] == "etf" else cfg.stock_weight_cap
        p["weight"] = min(cap, base)

    etf_weight = sum(p["weight"] for p in picks if p["asset_type"] == "etf")
    if etf_weight < cfg.min_etf_weight * gross and picks:
        scale = (cfg.min_etf_weight * gross) / max(etf_weight, 1e-9)
        scale = min(scale, 1.5)
        for p in picks:
            if p["asset_type"] == "etf":
                p["weight"] = min(cfg.etf_weight_cap, p["weight"] * scale)

    total = sum(p["weight"] for p in picks)
    if total > gross:
        f = gross / total
        for p in picks:
            p["weight"] *= f

    return picks


def apply_macro_trades(
    *,
    cfg: MacroConfig,
    targets: list[dict[str, Any]],
    theme_key: str,
    logger,
) -> tuple[Any, list[dict[str, Any]]]:
    """Rebalance macro portfolio toward targets; returns new state and trade action dicts."""
    fx = fetch_usdcad_last()
    state = load_portfolio_state(cfg.portfolio_state_path, initial_cash_cad=cfg.portfolio_budget_cad)
    event_path = resolve_portfolio_event_log_path(cfg.portfolio_state_path)
    equity = max(_equity_cad(state, fx), float(state.cash_cad) or cfg.portfolio_budget_cad)

    open_by_ticker = {p.ticker.upper(): p for p in _open_positions(state)}
    target_map = {t["ticker"].upper(): t for t in targets}

    actions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    now = _utcnow().isoformat()

    new_entries = 0
    full_exits = 0

    for tkr, pos in list(open_by_ticker.items()):
        if tkr in target_map:
            continue
        if pos.days_held() < cfg.min_hold_days:
            continue
        px = last_close_cad(tkr, fx_usdcad=fx)
        if px is None or px <= 0:
            continue
        if full_exits >= cfg.max_full_exits_per_run:
            break
        shares = float(pos.shares)
        events.append(
            {
                "action": "SELL",
                "ticker": tkr,
                "shares": shares,
                "price_cad": px,
                "ts_utc": now,
                "reason": "MACRO_EXIT_NOT_IN_TARGET",
            }
        )
        actions.append(
            {
                "ticker": tkr,
                "action": "EXIT",
                "reason": "not_in_macro_targets",
                "shares": shares,
                "price_cad": px,
            }
        )
        full_exits += 1

    for t in targets:
        tkr = t["ticker"].upper()
        w = float(t["weight"])
        notion = equity * w
        px = last_close_cad(tkr, fx_usdcad=fx)
        if px is None or px <= 0 or notion <= 0:
            continue
        target_shares = notion / px
        pos = open_by_ticker.get(tkr)
        cur_sh = float(pos.shares) if pos else 0.0
        delta_sh = target_shares - cur_sh
        if abs(delta_sh * px) < max(25.0, 0.02 * equity):
            continue

        if delta_sh > 0:
            if new_entries >= cfg.max_new_positions_per_run and not pos:
                continue
            buy_sh = delta_sh
            events.append(
                {
                    "action": "BUY",
                    "ticker": tkr,
                    "shares": buy_sh,
                    "price_cad": px,
                    "ts_utc": now,
                    "reason": "MACRO_ENTRY_OR_ADD",
                    "macro_theme_key": theme_key,
                    "macro_basket_key": t.get("basket_key"),
                    "macro_theme_cluster": t.get("theme_cluster"),
                }
            )
            actions.append(
                {
                    "ticker": tkr,
                    "action": "BUY",
                    "reason": "macro_target_add",
                    "shares": buy_sh,
                    "price_cad": px,
                }
            )
            if not pos:
                new_entries += 1
            continue

        if pos is None:
            continue
        if pos.days_held() < cfg.min_hold_days:
            continue
        sell_sh = min(float(pos.shares), abs(delta_sh))
        if sell_sh <= 0:
            continue
        act = "SELL_PARTIAL" if sell_sh < float(pos.shares) - 1e-6 else "SELL"
        events.append(
            {
                "action": act,
                "ticker": tkr,
                "shares": sell_sh,
                "price_cad": px,
                "ts_utc": now,
                "reason": "MACRO_TRIM",
            }
        )
        actions.append(
            {
                "ticker": tkr,
                "action": "TRIM" if act == "SELL_PARTIAL" else "EXIT",
                "reason": "macro_target_trim",
                "shares": sell_sh,
                "price_cad": px,
            }
        )

    if events and not cfg.dry_run:
        append_portfolio_events(event_path, events)
        state = load_portfolio_state(cfg.portfolio_state_path, initial_cash_cad=cfg.portfolio_budget_cad)

    state = replace(state, last_updated=_utcnow())
    if not cfg.dry_run:
        save_portfolio_state(cfg.portfolio_state_path, state)

    return state, actions

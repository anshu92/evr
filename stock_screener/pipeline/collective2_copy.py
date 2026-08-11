from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from stock_screener.agents.config import get_agent_config
from stock_screener.agents.trading_agent import _call_llm
from stock_screener.data.collective2 import (
    C2Signal,
    C2System,
    C2Trade,
    Collective2Client,
    is_long_stock_system,
    is_supported_long_stock_signal,
    signal_posted_at,
)
from stock_screener.utils import ensure_dir, read_json, sanitize_ticker, write_json


@dataclass(frozen=True)
class Collective2CopyConfig:
    api_key: str
    initial_cash_usd: float = 4000.0
    state_path: str = "collective2_usd_portfolio_state.json"
    cache_dir: str = "cache"
    reports_dir: str = "reports"
    max_roster_systems: int = 200
    max_ranked_systems: int = 25
    max_accessible_systems: int = 15
    max_candidate_signals: int = 20
    signal_lookback_minutes: int = 45
    stale_signal_minutes: int = 90
    c2_throttle_seconds: float = 1.0
    max_positions: int = 8
    max_ticker_weight: float = 0.20
    max_system_weight: float = 0.40
    cash_buffer_pct: float = 0.05
    min_trade_notional_usd: float = 50.0
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.12
    deterministic_accept_threshold: float = 0.58
    max_minimum_portfolio_usd: float = 100000.0
    mock_mode: bool = False

    @staticmethod
    def from_env() -> "Collective2CopyConfig":
        def _s(name: str, default: str) -> str:
            raw = os.getenv(name)
            return raw if raw not in {None, ""} else default

        def _i(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw not in {None, ""} else default

        def _f(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw not in {None, ""} else default

        def _b(name: str, default: bool) -> bool:
            raw = os.getenv(name).strip().lower() if os.getenv(name) else None
            if raw in ("1", "true", "yes"): return True
            if raw in ("0", "false", "no"): return False
            return default

        return Collective2CopyConfig(
            api_key=_s("COLLECTIVE2_API_KEY", ""),
            initial_cash_usd=_f("COLLECTIVE2_INITIAL_CASH_USD", 4000.0),
            state_path=_s("COLLECTIVE2_PORTFOLIO_STATE_PATH", "collective2_usd_portfolio_state.json"),
            cache_dir=_s("CACHE_DIR", "cache"),
            reports_dir=_s("REPORTS_DIR", "reports"),
            max_roster_systems=_i("COLLECTIVE2_MAX_ROSTER_SYSTEMS", 200),
            max_ranked_systems=_i("COLLECTIVE2_MAX_RANKED_SYSTEMS", 25),
            max_accessible_systems=_i("COLLECTIVE2_MAX_ACCESSIBLE_SYSTEMS", 15),
            max_candidate_signals=_i("COLLECTIVE2_MAX_CANDIDATE_SIGNALS", 20),
            signal_lookback_minutes=_i("COLLECTIVE2_SIGNAL_LOOKBACK_MINUTES", 45),
            stale_signal_minutes=_i("COLLECTIVE2_STALE_SIGNAL_MINUTES", 90),
            c2_throttle_seconds=_f("COLLECTIVE2_THROTTLE_SECONDS", 1.0),
            max_positions=_i("COLLECTIVE2_MAX_POSITIONS", 8),
            max_ticker_weight=_f("COLLECTIVE2_MAX_TICKER_WEIGHT", 0.20),
            max_system_weight=_f("COLLECTIVE2_MAX_SYSTEM_WEIGHT", 0.40),
            cash_buffer_pct=_f("COLLECTIVE2_CASH_BUFFER_PCT", 0.05),
            min_trade_notional_usd=_f("COLLECTIVE2_MIN_TRADE_NOTIONAL_USD", 50.0),
            stop_loss_pct=_f("COLLECTIVE2_STOP_LOSS_PCT", 0.08),
            take_profit_pct=_f("COLLECTIVE2_TAKE_PROFIT_PCT", 0.12),
            deterministic_accept_threshold=_f("COLLECTIVE2_ACCEPT_THRESHOLD", 0.58),
            max_minimum_portfolio_usd=_f("COLLECTIVE2_MAX_MINIMUM_PORTFOLIO_USD", 100000.0),
            mock_mode=_b("COLLECTIVE2_MOCK_MODE", False),
        )


@dataclass
class CopyPosition:
    ticker: str
    shares: float
    entry_price_usd: float
    entry_time_utc: str
    system_id: str
    system_name: str
    signal_id: str
    highest_price_usd: float | None = None


@dataclass
class CopyPortfolioState:
    cash_usd: float
    positions: list[CopyPosition] = field(default_factory=list)
    last_updated_utc: str = ""
    realized_pnl_usd: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SystemPerformance:
    system_id: str
    system_name: str
    owner_screenname: str
    minimum_portfolio_size_required: float
    closed_trade_count: int
    open_trade_count: int
    winning_trade_count: int
    win_rate: float
    total_pl: float
    average_pl: float
    performance_score: float
    raw_details: dict[str, Any]


@dataclass(frozen=True)
class SignalCandidate:
    signal: C2Signal
    system: C2System
    system_score: float
    deterministic_score: float
    latest_price_usd: float
    recent_trades: list[dict[str, Any]]
    performance: SystemPerformance
    source_trade_notional_usd: float | None
    source_trade_pct_of_portfolio: float | None
    consensus_count: int
    consensus_sources: list[dict[str, Any]]
    reason: str


@dataclass(frozen=True)
class SignalDecision:
    signal_id: str
    system_id: str
    ticker: str
    decision: str
    target_weight: float
    confidence: float
    reason: str
    explanation: str
    risk_flags: list[str]
    deterministic_score: float
    llm_used: bool
    source_system_name: str = ""
    source_owner: str = ""
    source_trade_pct_of_portfolio: float | None = None
    consensus_count: int = 1


def run_collective2_copy(cfg: Collective2CopyConfig, logger) -> None:
    started = datetime.now(tz=timezone.utc)
    cache_dir = ensure_dir(cfg.cache_dir)
    reports_dir = ensure_dir(cfg.reports_dir)
    _write_email_gate(reports_dir, started)
    if not cfg.api_key.strip():
        _write_no_key_reports(cfg, reports_dir, logger)
        return

    client = Collective2Client(
        api_key=cfg.api_key,
        throttle_seconds=cfg.c2_throttle_seconds,
    )
    state_path = Path(cfg.state_path)
    state = load_copy_state(state_path, initial_cash_usd=cfg.initial_cash_usd)
    seen_path = cache_dir / "collective2_seen_signals.json"
    seen = _load_seen(seen_path)

    roster = client.get_system_roster(filter_value="active")[: cfg.max_roster_systems]
    eligible = [
        s for s in roster
        if is_long_stock_system(s, max_minimum_portfolio_usd=cfg.max_minimum_portfolio_usd)
    ]
    access_rows = client.list_all_systems()
    accessible_ids = {_system_id_from_row(x) for x in access_rows}
    accessible_ids.discard("")
    
    if cfg.mock_mode and not accessible_ids:
        logger.warning("MOCK_MODE: listAllSystems is empty; simulating access to top-ranked systems")
        # Treat top 5 eligible systems as accessible for testing
        accessible_ids = {s.system_id for s in eligible[:5]}

    ranked = rank_systems(eligible)
    write_json(cache_dir / "collective2_access_audit.json", {
        "run_utc": started.isoformat(),
        "roster_count": len(roster),
        "eligible_count": len(eligible),
        "accessible_count": len(accessible_ids),
        "accessible_system_ids": sorted(accessible_ids),
        "roster_rejection_counts": _roster_rejection_counts(roster, cfg),
        "roster_sample": [_system_audit_sample(s) for s in roster[:10]],
        "access_row_count": len(access_rows),
        "access_row_sample": access_rows[:5],
    })

    ranked_payload: list[dict[str, Any]] = []
    details_by_system: dict[str, dict[str, Any]] = {}
    trades_by_system: dict[str, list[dict[str, Any]]] = {}
    open_trades_by_system: dict[str, list[dict[str, Any]]] = {}
    accessible_ranked = [x for x in ranked if x[0].system_id in accessible_ids][: cfg.max_accessible_systems]
    for system, score in ranked[: cfg.max_ranked_systems]:
        details: dict[str, Any] = {}
        if system.system_id in accessible_ids or len(details_by_system) < min(5, cfg.max_ranked_systems):
            try:
                details = client.get_system_details(system.system_id)
            except Exception as e:
                logger.warning("Could not fetch C2 details for %s: %s", system.system_id, e)
        details_by_system[system.system_id] = details
        ranked_payload.append({
            "system_id": system.system_id,
            "system_name": system.system_name,
            "owner_screenname": system.owner_screenname,
            "accessible": system.system_id in accessible_ids,
            "score": score,
            "minimum_portfolio_size_required": system.minimum_portfolio_size_required,
            "monthly_fee": system.monthly_fee,
            "free_trial_days": system.free_trial_days,
            "created_when": system.created_when,
        })
    write_json(cache_dir / "collective2_system_rankings.json", {
        "run_utc": started.isoformat(),
        "systems": ranked_payload,
    })
    _write_public_roster_report(
        reports_dir,
        started=started,
        access_summary={
            "roster_count": len(roster),
            "eligible_count": len(eligible),
            "accessible_count": len(accessible_ids),
            "accessible_ranked_count": len(accessible_ranked),
        },
        ranked_payload=ranked_payload,
        audit={
            "roster_rejection_counts": _roster_rejection_counts(roster, cfg),
            "access_row_count": len(access_rows),
        },
    )

    raw_signals: list[C2Signal] = []
    start_ny, end_ny = _ny_time_window(cfg.signal_lookback_minutes)
    systems_by_id = {s.system_id: s for s, _ in ranked}
    scores_by_id = {s.system_id: score for s, score in ranked}
    
    # Ensure we fetch open trades for EVERY system we currently hold, 
    # even if it dropped out of the top ranked accessible list.
    held_system_ids = {p.system_id for p in state.positions}
    fetch_targets = list(accessible_ranked)
    for sid in held_system_ids:
        if sid not in {s.system_id for s, _ in accessible_ranked} and sid in accessible_ids:
            # Re-fetch metadata if we can, otherwise use a placeholder
            sys_obj = systems_by_id.get(sid)
            if sys_obj:
                fetch_targets.append((sys_obj, scores_by_id.get(sid, 0.5)))

    for system, _score in fetch_targets:
        try:
            # For the roster-diff approach, we still poll signals to catch 
            # intraday events quickly, but requestTradesOpen is the source of truth.
            raw_signals.extend(client.retrieve_signals_all(
                system.system_id,
                filter_type="time_posted",
                start_ny=start_ny,
                end_ny=end_ny,
            ))
            raw_signals.extend(client.retrieve_signals_working(system.system_id))
            trades = client.request_trades(system.system_id, open_only=False)[:50]
            open_trades = client.request_trades(system.system_id, open_only=True)[:50]
            trades_by_system[system.system_id] = [asdict(t) for t in trades]
            open_trades_by_system[system.system_id] = [asdict(t) for t in open_trades]
        except Exception as e:
            if cfg.mock_mode:
                logger.info("MOCK_MODE: Injecting fake trades/signals for system %s", system.system_id)
                fake_trades = _generate_mock_trades(system.system_id)
                open_trades_by_system[system.system_id] = [asdict(t) for t in fake_trades if t.open_or_closed == "open"]
                trades_by_system[system.system_id] = [asdict(t) for t in fake_trades]
            else:
                logger.warning("Could not poll C2 signals for %s: %s", system.system_id, e)

    # Infer BTO signals from current rosters (primary source)
    inferred_signals = infer_signals_from_trades(
        open_trades_by_system,
        state,
        max_age_minutes=cfg.stale_signal_minutes,
        logger=logger,
    )
    
    # We still use polled signals as a supplement (may catch very fresh signals 
    # that haven't appeared in requestTradesOpen cache yet).
    polled_signals = normalize_signals(raw_signals, seen, max_age_minutes=cfg.stale_signal_minutes)
    
    # Combine signals, prioritizing polled (usually have more detail) over inferred
    combined_signals_dict: dict[str, C2Signal] = {s.signal_id: s for s in inferred_signals}
    for s in polled_signals:
        combined_signals_dict[s.signal_id] = s
    normalized = sorted(combined_signals_dict.values(), key=lambda s: s.posted_time_unix, reverse=True)

    tickers = sorted({s.symbol for s in normalized} | {p.ticker for p in state.positions})
    latest_prices = fetch_latest_usd_prices(tickers, logger=logger)

    # Hard Roster Sync: If we hold it but the source system doesn't, we exit.
    sync_events = apply_roster_sync_exits(state, open_trades_by_system, latest_prices, started, logger)
    
    stop_events = apply_mechanical_exits(state, latest_prices, cfg, started)
    performance_by_system = build_performance_profiles(
        systems_by_id=systems_by_id,
        trades_by_system=trades_by_system,
        open_trades_by_system=open_trades_by_system,
        details_by_system=details_by_system,
    )

    candidates = build_candidates(
        normalized,
        systems_by_id=systems_by_id,
        scores_by_id=scores_by_id,
        latest_prices=latest_prices,
        trades_by_system=trades_by_system,
        performance_by_system=performance_by_system,
        max_candidates=cfg.max_candidate_signals,
    )
    decisions = select_with_llm_or_rules(candidates, state, latest_prices, cfg, logger)
    trade_events = apply_decisions(state, decisions, candidates, latest_prices, cfg, started)
    all_events = sync_events + stop_events + trade_events
    for sig in normalized:
        seen.add(sig.signal_id)
    _save_seen(seen_path, seen)

    state.last_updated_utc = started.isoformat()
    save_copy_state(state_path, state)
    append_copy_events(state_path, all_events)
    render_reports(
        reports_dir,
        cfg=cfg,
        state=state,
        latest_prices=latest_prices,
        decisions=decisions,
        candidates=candidates,
        access_summary={
            "roster_count": len(roster),
            "eligible_count": len(eligible),
            "accessible_count": len(accessible_ids),
            "accessible_ranked_count": len(accessible_ranked),
        },
        events=all_events,
        started=started,
    )
    logger.info("Collective2 copy run finished: candidates=%d decisions=%d events=%d", len(candidates), len(decisions), len(all_events))


def rank_systems(systems: list[C2System]) -> list[tuple[C2System, float]]:
    ranked: list[tuple[C2System, float]] = []
    now_year = datetime.now(tz=timezone.utc).year
    for s in systems:
        age_score = 0.0
        if s.created_when[:4].isdigit():
            age_score = min(0.25, max(0.0, (now_year - int(s.created_when[:4])) / 40.0))
        min_score = 0.25 if s.minimum_portfolio_size_required <= 4000 else 0.0
        fee_score = 0.15 if s.monthly_fee <= 50 else 0.05 if s.monthly_fee <= 150 else 0.0
        trial_score = 0.10 if s.free_trial_days > 0 else 0.0
        score = 0.50 + age_score + min_score + fee_score + trial_score
        ranked.append((s, min(1.0, score)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _system_id_from_row(row: dict[str, Any]) -> str:
    return str(
        row.get("system_id")
        or row.get("systemid")
        or row.get("systemId")
        or row.get("id")
        or row.get("c2systemid")
        or ""
    ).strip()


def _roster_rejection_counts(roster: list[C2System], cfg: Collective2CopyConfig) -> dict[str, int]:
    counts = {
        "missing_system_id": 0,
        "not_alive": 0,
        "not_stock_capable": 0,
        "has_stock_short_flag": 0,
        "eligible": 0,
    }
    for system in roster:
        reason = _system_rejection_reason(system, cfg)
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _system_rejection_reason(system: C2System, cfg: Collective2CopyConfig) -> str:
    if not system.system_id:
        return "missing_system_id"
    if not system.is_alive:
        return "not_alive"
    if not system.trades_stocks:
        return "not_stock_capable"
    if system.trades_stocks_short:
        return "has_stock_short_flag"
    return "eligible"


def _system_audit_sample(system: C2System) -> dict[str, Any]:
    return {
        "system_id": system.system_id,
        "system_name": system.system_name,
        "owner_screenname": system.owner_screenname,
        "trades_stocks": system.trades_stocks,
        "trades_stocks_short": system.trades_stocks_short,
        "trades_options": system.trades_options,
        "trades_futures": system.trades_futures,
        "trades_forex": system.trades_forex,
        "minimum_portfolio_size_required": system.minimum_portfolio_size_required,
        "is_alive": system.is_alive,
        "raw_keys": sorted(system.raw.keys())[:40],
    }


def apply_roster_sync_exits(
    state: CopyPortfolioState,
    open_trades_by_system: dict[str, list[dict[str, Any]]],
    latest_prices: dict[str, float],
    now: datetime,
    logger,
) -> list[dict[str, Any]]:
    """Exit positions that are no longer present in the source system's roster."""
    events: list[dict[str, Any]] = []
    sync_count = 0
    
    # We only sync systems for which we successfully fetched the current roster
    active_roster_system_ids = set(open_trades_by_system.keys())
    
    for pos in list(state.positions):
        if pos.system_id not in active_roster_system_ids:
            continue
            
        system_roster = open_trades_by_system[pos.system_id]
        is_held_by_source = any(
            str(t.get("symbol") or "").strip().upper() == pos.ticker.upper()
            for t in system_roster
        )
        
        if not is_held_by_source:
            px = latest_prices.get(pos.ticker, pos.entry_price_usd)
            logger.info("ROSTER SYNC: Exiting %s (no longer held by source system %s)", pos.ticker, pos.system_id)
            events.extend(_sell_position(state, pos, float(px), "ROSTER_SYNC_EXIT", now))
            sync_count += 1
            
    if sync_count > 0:
        logger.warning("ROSTER SYNC: closed %d position(s) that were missing from source rosters", sync_count)
    return events


def infer_signals_from_trades(
    open_trades_by_system: dict[str, list[dict[str, Any]]],
    state: CopyPortfolioState,
    *,
    max_age_minutes: int,
    logger,
) -> list[C2Signal]:
    """Synthesize BTO signals from recently opened trades in source rosters."""
    inferred: list[C2Signal] = []
    now = datetime.now(tz=timezone.utc)
    
    # Track what we already hold to avoid duplicate BUY signals
    held_tickers_by_system = {(p.ticker.upper(), p.system_id) for p in state.positions}
    
    for system_id, trades in open_trades_by_system.items():
        for t in trades:
            symbol = str(t.get("symbol") or "").strip().upper()
            if not symbol:
                continue
                
            # Skip if we already hold it from this system
            if (symbol, system_id) in held_tickers_by_system:
                continue
                
            # Filter for long stocks only
            instrument = str(t.get("instrument") or "").lower()
            if instrument not in {"", "stock"}:
                continue
            if str(t.get("long_or_short") or "").lower() != "long":
                continue
                
            # Check if it was opened recently
            opened_at_str = str(t.get("opened_when") or "")
            opened_at: datetime | None = None
            if opened_at_str:
                try:
                    # C2 usually provides "YYYY-MM-DD HH:MM:SS" in New York time
                    # We treat it as UTC if no TZ is provided, or try to parse
                    opened_at = datetime.fromisoformat(opened_at_str.replace(" ", "T"))
                    if opened_at.tzinfo is None:
                        # Assume NY time as that is C2's default, but for robustness
                        # we compare using a generous window.
                        opened_at = opened_at.replace(tzinfo=ZoneInfo("America/New_York"))
                except Exception:
                    logger.warning("Could not parse opened_when for trade %s: %s", t.get("trade_id"), opened_at_str)
            
            if opened_at is None:
                continue
                
            # Normalize to UTC for reliable age comparison
            opened_at_utc = opened_at.astimezone(timezone.utc)
            age_seconds = (now - opened_at_utc).total_seconds()
            if age_seconds > max_age_minutes * 60:
                continue
                
            # Synthesize a C2Signal
            inferred.append(C2Signal(
                system_id=system_id,
                signal_id=f"inferred_{t.get('trade_id')}_{int(opened_at_utc.timestamp())}",
                symbol=symbol,
                action="BTO",
                quantity=float(t.get("quantity") or 0.0),
                status="filled",
                instrument=instrument,
                posted_time=opened_at_utc.isoformat(),
                posted_time_unix=int(opened_at_utc.timestamp()),
                traded_time_unix=int(opened_at_utc.timestamp()),
                is_market_order=True,
                is_limit_order=False,
                is_stop_order=False,
                raw=dict(t),
            ))
            
    if inferred:
        logger.info("ROSTER SYNC: Inferred %d BTO signal(s) from current rosters", len(inferred))
    return inferred


def normalize_signals(raw_signals: list[C2Signal], seen: set[str], *, max_age_minutes: int) -> list[C2Signal]:
    now = datetime.now(tz=timezone.utc)
    out: dict[str, C2Signal] = {}
    for sig in raw_signals:
        if sig.signal_id in seen:
            continue
        if not is_supported_long_stock_signal(sig):
            continue
        posted = signal_posted_at(sig)
        if posted is not None and (now - posted).total_seconds() > max_age_minutes * 60:
            continue
        out[sig.signal_id] = sig
    return sorted(out.values(), key=lambda s: s.posted_time_unix, reverse=True)


def build_performance_profiles(
    *,
    systems_by_id: dict[str, C2System],
    trades_by_system: dict[str, list[dict[str, Any]]],
    open_trades_by_system: dict[str, list[dict[str, Any]]],
    details_by_system: dict[str, dict[str, Any]],
) -> dict[str, SystemPerformance]:
    profiles: dict[str, SystemPerformance] = {}
    for system_id, system in systems_by_id.items():
        trades = trades_by_system.get(system_id, [])
        open_trades = open_trades_by_system.get(system_id, [])
        closed = [
            t for t in trades
            if str(t.get("open_or_closed") or "").lower() in {"closed", "close", ""}
        ]
        pls = [_float(t.get("pl"), 0.0) for t in closed]
        wins = sum(1 for x in pls if x > 0)
        total_pl = float(sum(pls))
        closed_count = len(closed)
        win_rate = float(wins / closed_count) if closed_count else 0.0
        average_pl = float(total_pl / closed_count) if closed_count else 0.0
        sample_score = min(0.20, closed_count / 100.0)
        win_score = max(0.0, min(0.35, (win_rate - 0.45) * 0.70 + 0.15))
        pl_score = 0.25 if total_pl > 0 else 0.10 if closed_count > 0 else 0.0
        score = max(0.0, min(1.0, 0.20 + sample_score + win_score + pl_score))
        profiles[system_id] = SystemPerformance(
            system_id=system_id,
            system_name=system.system_name,
            owner_screenname=system.owner_screenname,
            minimum_portfolio_size_required=float(system.minimum_portfolio_size_required or 0.0),
            closed_trade_count=closed_count,
            open_trade_count=len(open_trades),
            winning_trade_count=wins,
            win_rate=win_rate,
            total_pl=total_pl,
            average_pl=average_pl,
            performance_score=score,
            raw_details=details_by_system.get(system_id, {}),
        )
    return profiles


def build_candidates(
    signals: list[C2Signal],
    *,
    systems_by_id: dict[str, C2System],
    scores_by_id: dict[str, float],
    latest_prices: dict[str, float],
    trades_by_system: dict[str, list[dict[str, Any]]],
    performance_by_system: dict[str, SystemPerformance] | None = None,
    max_candidates: int,
) -> list[SignalCandidate]:
    out: list[SignalCandidate] = []
    performance_by_system = performance_by_system or {}
    consensus = _build_signal_consensus(signals, systems_by_id, performance_by_system)
    for sig in signals:
        system = systems_by_id.get(sig.system_id)
        px = latest_prices.get(sig.symbol)
        if system is None or px is None or px <= 0:
            continue
        performance = performance_by_system.get(sig.system_id) or _empty_performance(system)
        source_notional = float(sig.quantity) * float(px) if sig.quantity > 0 else None
        source_pct = None
        if source_notional is not None and performance.minimum_portfolio_size_required > 0:
            source_pct = source_notional / performance.minimum_portfolio_size_required
        consensus_sources = consensus.get((sig.symbol, sig.action), [])
        consensus_count = max(1, len({str(x.get("system_id")) for x in consensus_sources}))
        trade_size_score = 0.50 if source_pct is None else max(0.0, min(1.0, source_pct / 0.10))
        consensus_score = min(0.15, (consensus_count - 1) * 0.05)
        score = (
            scores_by_id.get(sig.system_id, 0.5) * 0.35
            + performance.performance_score * 0.35
            + trade_size_score * 0.15
            + 0.10
            + consensus_score
        )
        if sig.is_market_order:
            score += 0.05
        if sig.action == "BTO":
            score += 0.05
        if sig.action == "STC":
            score += 0.02
        score = max(0.0, min(1.0, score))
        out.append(SignalCandidate(
            signal=sig,
            system=system,
            system_score=scores_by_id.get(sig.system_id, 0.5),
            deterministic_score=score,
            latest_price_usd=float(px),
            recent_trades=trades_by_system.get(sig.system_id, [])[:20],
            performance=performance,
            source_trade_notional_usd=source_notional,
            source_trade_pct_of_portfolio=source_pct,
            consensus_count=consensus_count,
            consensus_sources=consensus_sources,
            reason="fresh long stock/ETF signal from accessible C2 system",
        ))
    out.sort(key=lambda c: c.deterministic_score, reverse=True)
    return out[:max_candidates]


def _empty_performance(system: C2System) -> SystemPerformance:
    return SystemPerformance(
        system_id=system.system_id,
        system_name=system.system_name,
        owner_screenname=system.owner_screenname,
        minimum_portfolio_size_required=float(system.minimum_portfolio_size_required or 0.0),
        closed_trade_count=0,
        open_trade_count=0,
        winning_trade_count=0,
        win_rate=0.0,
        total_pl=0.0,
        average_pl=0.0,
        performance_score=0.35,
        raw_details={},
    )


def _build_signal_consensus(
    signals: list[C2Signal],
    systems_by_id: dict[str, C2System],
    performance_by_system: dict[str, SystemPerformance],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for sig in signals:
        system = systems_by_id.get(sig.system_id)
        performance = performance_by_system.get(sig.system_id)
        if system is None:
            continue
        if performance is not None and performance.closed_trade_count > 0 and performance.total_pl <= 0:
            continue
        key = (sig.symbol, sig.action)
        grouped.setdefault(key, {})[sig.system_id] = {
            "system_id": sig.system_id,
            "system_name": system.system_name,
            "owner_screenname": system.owner_screenname,
            "performance_score": performance.performance_score if performance else None,
            "win_rate": performance.win_rate if performance else None,
            "total_pl": performance.total_pl if performance else None,
        }
    return {k: list(v.values()) for k, v in grouped.items()}


def select_with_llm_or_rules(
    candidates: list[SignalCandidate],
    state: CopyPortfolioState,
    latest_prices: dict[str, float],
    cfg: Collective2CopyConfig,
    logger,
) -> list[SignalDecision]:
    if not candidates:
        return []
    agent_cfg = get_agent_config()
    if not agent_cfg.get("model_chain") and not agent_cfg.get("smart_chain"):
        return [
            _rule_decision(c, cfg, llm_used=False)
            for c in candidates
            if c.deterministic_score >= cfg.deterministic_accept_threshold or c.signal.action == "STC"
        ]

    prompt = _llm_prompt(candidates, state, latest_prices)
    raw = _call_llm(
        None,
        agent_cfg,
        "You select paper copy-trade signals. Return only JSON.",
        prompt,
        max_tokens_override=1200,
        model_override=agent_cfg.get("smart_model") or "smart",
    )
    parsed = _parse_llm_decisions(raw)
    if not parsed:
        logger.warning("Collective2 LLM returned no parseable decisions; falling back to deterministic scoring")
        return [
            _rule_decision(c, cfg, llm_used=False)
            for c in candidates
            if c.deterministic_score >= cfg.deterministic_accept_threshold or c.signal.action == "STC"
        ]

    by_key = {(d["system_id"], d["signal_id"]): d for d in parsed}
    decisions: list[SignalDecision] = []
    for c in candidates:
        d = by_key.get((c.signal.system_id, c.signal.signal_id))
        if not d:
            continue
        decision = str(d.get("decision", "REJECT")).strip().upper()
        if decision not in {"ACCEPT", "REJECT", "EXIT"}:
            decision = "REJECT"
        decisions.append(SignalDecision(
            signal_id=c.signal.signal_id,
            system_id=c.signal.system_id,
            ticker=c.signal.symbol,
            decision=decision,
            target_weight=_scaled_target_weight(c, _float(d.get("target_weight"), 0.0), cfg),
            confidence=max(0.0, min(1.0, _float(d.get("confidence"), 0.0))),
            reason=str(d.get("reason") or "").strip()[:500],
            explanation=str(d.get("explanation") or d.get("reason") or "").strip()[:1500],
            risk_flags=[str(x)[:120] for x in d.get("risk_flags", []) if isinstance(x, (str, int, float))] if isinstance(d.get("risk_flags"), list) else [],
            deterministic_score=c.deterministic_score,
            llm_used=True,
            source_system_name=c.system.system_name,
            source_owner=c.system.owner_screenname,
            source_trade_pct_of_portfolio=c.source_trade_pct_of_portfolio,
            consensus_count=c.consensus_count,
        ))
    return decisions


def _rule_decision(candidate: SignalCandidate, cfg: Collective2CopyConfig, *, llm_used: bool) -> SignalDecision:
    action = candidate.signal.action
    decision = "EXIT" if action == "STC" else "ACCEPT"
    fallback_weight = max(cfg.min_trade_notional_usd / cfg.initial_cash_usd, candidate.deterministic_score * 0.20)
    weight = _scaled_target_weight(candidate, fallback_weight, cfg)
    return SignalDecision(
        signal_id=candidate.signal.signal_id,
        system_id=candidate.signal.system_id,
        ticker=candidate.signal.symbol,
        decision=decision,
        target_weight=weight,
        confidence=candidate.deterministic_score,
        reason="deterministic fallback decision",
        explanation=(
            f"Deterministic score {candidate.deterministic_score:.2f}; "
            f"source win rate {candidate.performance.win_rate:.1%}; "
            f"same-ticker source count {candidate.consensus_count}."
        ),
        risk_flags=[],
        deterministic_score=candidate.deterministic_score,
        llm_used=llm_used,
        source_system_name=candidate.system.system_name,
        source_owner=candidate.system.owner_screenname,
        source_trade_pct_of_portfolio=candidate.source_trade_pct_of_portfolio,
        consensus_count=candidate.consensus_count,
    )


def _scaled_target_weight(candidate: SignalCandidate, requested_weight: float, cfg: Collective2CopyConfig) -> float:
    source_weight = candidate.source_trade_pct_of_portfolio
    if source_weight is not None and source_weight > 0:
        return max(0.0, min(cfg.max_ticker_weight, float(source_weight)))
    return max(0.0, min(cfg.max_ticker_weight, float(requested_weight)))


def apply_decisions(
    state: CopyPortfolioState,
    decisions: list[SignalDecision],
    candidates: list[SignalCandidate],
    latest_prices: dict[str, float],
    cfg: Collective2CopyConfig,
    now: datetime,
) -> list[dict[str, Any]]:
    by_signal = {c.signal.signal_id: c for c in candidates}
    events: list[dict[str, Any]] = []
    equity = compute_equity_usd(state, latest_prices)
    for d in decisions:
        c = by_signal.get(d.signal_id)
        if c is None:
            continue
        px = latest_prices.get(d.ticker)
        if px is None or px <= 0:
            continue
        if d.decision == "EXIT":
            events.extend(_sell_matching(state, d.ticker, float(px), f"C2_EXIT:{d.reason}", now))
            continue
        if d.decision != "ACCEPT" or c.signal.action != "BTO":
            continue
        if any(p.ticker == d.ticker for p in state.positions):
            continue
        if len(state.positions) >= cfg.max_positions:
            continue
        system_exposure = _system_exposure(state, latest_prices, d.system_id)
        max_system_notional = equity * cfg.max_system_weight
        if system_exposure >= max_system_notional:
            continue
        target_notional = min(equity * d.target_weight, max_system_notional - system_exposure)
        available_cash = max(0.0, state.cash_usd - equity * cfg.cash_buffer_pct)
        notional = min(target_notional, available_cash)
        if notional < cfg.min_trade_notional_usd:
            continue
        shares = notional / float(px)
        state.cash_usd -= notional
        state.positions.append(CopyPosition(
            ticker=d.ticker,
            shares=shares,
            entry_price_usd=float(px),
            entry_time_utc=now.isoformat(),
            system_id=d.system_id,
            system_name=c.system.system_name,
            signal_id=d.signal_id,
            highest_price_usd=float(px),
        ))
        events.append({
            "ts_utc": now.isoformat(),
            "action": "BUY",
            "ticker": d.ticker,
            "shares": shares,
            "price_usd": float(px),
            "notional_usd": notional,
            "system_id": d.system_id,
            "system_name": c.system.system_name,
            "source_owner": c.system.owner_screenname,
            "signal_id": d.signal_id,
            "reason": d.reason,
            "explanation": d.explanation,
            "confidence": d.confidence,
            "target_weight": d.target_weight,
            "source_trade_quantity": c.signal.quantity,
            "source_trade_notional_usd": c.source_trade_notional_usd,
            "source_trade_pct_of_portfolio": c.source_trade_pct_of_portfolio,
            "source_performance": asdict(c.performance),
            "consensus_count": c.consensus_count,
            "consensus_sources": c.consensus_sources,
            "paper_only": True,
        })
    return events


def apply_mechanical_exits(
    state: CopyPortfolioState,
    latest_prices: dict[str, float],
    cfg: Collective2CopyConfig,
    now: datetime,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for pos in list(state.positions):
        px = latest_prices.get(pos.ticker)
        if px is None or px <= 0:
            continue
        pos.highest_price_usd = max(float(pos.highest_price_usd or pos.entry_price_usd), float(px))
        ret = float(px) / float(pos.entry_price_usd) - 1.0
        if ret <= -abs(cfg.stop_loss_pct):
            events.extend(_sell_position(state, pos, float(px), "STOP_LOSS", now))
        elif ret >= abs(cfg.take_profit_pct):
            events.extend(_sell_position(state, pos, float(px), "TAKE_PROFIT", now))
    return events


def _sell_matching(state: CopyPortfolioState, ticker: str, price: float, reason: str, now: datetime) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for pos in list(state.positions):
        if pos.ticker == ticker:
            events.extend(_sell_position(state, pos, price, reason, now))
    return events


def _sell_position(state: CopyPortfolioState, pos: CopyPosition, price: float, reason: str, now: datetime) -> list[dict[str, Any]]:
    notional = float(pos.shares) * float(price)
    pnl = (float(price) - float(pos.entry_price_usd)) * float(pos.shares)
    state.cash_usd += notional
    state.realized_pnl_usd += pnl
    state.positions = [p for p in state.positions if p is not pos]
    return [{
        "ts_utc": now.isoformat(),
        "action": "SELL",
        "ticker": pos.ticker,
        "shares": pos.shares,
        "price_usd": float(price),
        "notional_usd": notional,
        "realized_pnl_usd": pnl,
        "system_id": pos.system_id,
        "signal_id": pos.signal_id,
        "reason": reason,
        "paper_only": True,
    }]


def compute_equity_usd(state: CopyPortfolioState, latest_prices: dict[str, float]) -> float:
    equity = float(state.cash_usd)
    for p in state.positions:
        px = latest_prices.get(p.ticker, p.entry_price_usd)
        equity += float(p.shares) * float(px)
    return float(equity)


def fetch_latest_usd_prices(tickers: list[str], *, logger=None) -> dict[str, float]:
    clean = [t for t in (sanitize_ticker(x) for x in tickers) if t]
    clean = list(dict.fromkeys(clean))
    if not clean:
        return {}
    try:
        df = yf.download(" ".join(clean), period="5d", interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        if logger:
            logger.warning("Price fetch failed for Collective2 tickers: %s", e)
        return {}
    prices: dict[str, float] = {}
    if df is None or df.empty:
        return prices
    if not isinstance(df.columns, pd.MultiIndex):
        close = df.get("Close")
        if close is not None and not close.dropna().empty:
            prices[clean[0]] = float(close.dropna().iloc[-1])
        return prices
    fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    if set(df.columns.get_level_values(0)).issubset(fields):
        df = df.swaplevel(0, 1, axis=1)
    for t in clean:
        try:
            close = df[(t, "Close")].dropna()
            if not close.empty and math.isfinite(float(close.iloc[-1])):
                prices[t] = float(close.iloc[-1])
        except Exception:
            continue
    return prices


def load_copy_state(path: Path, *, initial_cash_usd: float) -> CopyPortfolioState:
    if not path.exists():
        return CopyPortfolioState(cash_usd=float(initial_cash_usd), last_updated_utc=datetime.now(tz=timezone.utc).isoformat())
    data = read_json(path)
    positions = [CopyPosition(**p) for p in data.get("positions", []) if isinstance(p, dict)]
    return CopyPortfolioState(
        cash_usd=float(data.get("cash_usd", initial_cash_usd)),
        positions=positions,
        last_updated_utc=str(data.get("last_updated_utc") or ""),
        realized_pnl_usd=float(data.get("realized_pnl_usd", 0.0)),
        events=list(data.get("events", [])) if isinstance(data.get("events"), list) else [],
    )


def save_copy_state(path: Path, state: CopyPortfolioState) -> None:
    write_json(path, {
        "cash_usd": state.cash_usd,
        "positions": [asdict(p) for p in state.positions],
        "last_updated_utc": state.last_updated_utc,
        "realized_pnl_usd": state.realized_pnl_usd,
        "events": state.events[-200:],
    })


def append_copy_events(state_path: Path, events: list[dict[str, Any]]) -> None:
    if not events:
        return
    event_path = Path(str(state_path) + ".events.jsonl")
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, sort_keys=True) + "\n")


def render_reports(
    reports_dir: Path,
    *,
    cfg: Collective2CopyConfig,
    state: CopyPortfolioState,
    latest_prices: dict[str, float],
    decisions: list[SignalDecision],
    candidates: list[SignalCandidate],
    access_summary: dict[str, Any],
    events: list[dict[str, Any]],
    started: datetime,
) -> None:
    equity = compute_equity_usd(state, latest_prices)
    portfolio = {
        "run_utc": started.isoformat(),
        "cash_usd": state.cash_usd,
        "equity_usd": equity,
        "realized_pnl_usd": state.realized_pnl_usd,
        "positions": [asdict(p) for p in state.positions],
    }
    trades = {
        "run_utc": started.isoformat(),
        "access_summary": access_summary,
        "decisions": [asdict(d) for d in decisions],
        "candidates": [_candidate_report(c) for c in candidates],
        "candidate_count": len(candidates),
        "events": events,
        "paper_only": True,
    }
    write_json(reports_dir / "collective2_copy_portfolio.json", portfolio)
    write_json(reports_dir / "collective2_copy_trades.json", trades)

    lines = [
        "COLLECTIVE2 COPY-TRADE PAPER PORTFOLIO (USD)",
        f"Run UTC: {started.isoformat()}",
        f"Paper only: yes",
        f"Equity USD: ${equity:,.2f}",
        f"Cash USD: ${state.cash_usd:,.2f}",
        f"Realized PnL USD: ${state.realized_pnl_usd:,.2f}",
        f"Open positions: {len(state.positions)} / {cfg.max_positions}",
        f"Access: roster={access_summary.get('roster_count')} eligible={access_summary.get('eligible_count')} accessible={access_summary.get('accessible_count')}",
        "",
        "EVENTS",
    ]
    if events:
        for e in events:
            source = e.get("system_name") or e.get("system_id") or "unknown C2 system"
            confidence = e.get("confidence")
            conf_text = f" conf={float(confidence):.2f}" if isinstance(confidence, (int, float)) else ""
            lines.append(f"- {e['action']} {e['ticker']} ${e['notional_usd']:,.2f} @ ${e['price_usd']:.2f} from {source}{conf_text} ({e['reason']})")
    else:
        lines.append("- No paper trades this run.")
    lines.extend(["", "DECISIONS"])
    for d in decisions[:20]:
        lines.append(f"- {d.decision} {d.ticker} weight={d.target_weight:.1%} conf={d.confidence:.2f} consensus={d.consensus_count} source={d.source_system_name} llm={d.llm_used}: {d.reason}")
    lines.extend(["", "CANDIDATE SOURCES"])
    for c in candidates[:20]:
        pct = "unknown" if c.source_trade_pct_of_portfolio is None else f"{c.source_trade_pct_of_portfolio:.1%}"
        lines.append(
            f"- {c.signal.action} {c.signal.symbol} from {c.system.system_name} "
            f"win={c.performance.win_rate:.1%} pnl=${c.performance.total_pl:,.2f} "
            f"source_trade_pct={pct} consensus={c.consensus_count}"
        )
    text = "\n".join(lines) + "\n"
    (reports_dir / "collective2_copy_report.txt").write_text(text, encoding="utf-8")
    html = _render_email_html(
        state=state,
        equity=equity,
        cfg=cfg,
        started=started,
        events=events,
        decisions=decisions,
        candidates=candidates,
        access_summary=access_summary,
    )
    (reports_dir / "collective2_copy_email.html").write_text(html, encoding="utf-8")


def _write_public_roster_report(
    reports_dir: Path,
    *,
    started: datetime,
    access_summary: dict[str, Any],
    ranked_payload: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    lines = [
        "COLLECTIVE2 PUBLIC ROSTER RESEARCH",
        f"Run UTC: {started.isoformat()}",
        f"Roster count: {access_summary.get('roster_count')}",
        f"Eligible count: {access_summary.get('eligible_count')}",
        f"Accessible count: {access_summary.get('accessible_count')}",
        f"Accessible ranked count: {access_summary.get('accessible_ranked_count')}",
        f"Access row count: {audit.get('access_row_count')}",
        "",
        "REJECTION COUNTS",
    ]
    for key, value in sorted((audit.get("roster_rejection_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "TOP RANKED PUBLIC SYSTEMS"])
    if ranked_payload:
        for row in ranked_payload[:50]:
            lines.append(
                f"- {row.get('system_name')} ({row.get('system_id')}) "
                f"score={float(row.get('score') or 0.0):.3f} "
                f"stocks={row.get('minimum_portfolio_size_required')} "
                f"trial={row.get('free_trial_days')} fee=${float(row.get('monthly_fee') or 0.0):.2f} "
                f"accessible={row.get('accessible')}"
            )
    else:
        lines.append("- No eligible systems found.")
    text = "\n".join(lines) + "\n"
    (reports_dir / "collective2_public_roster_report.txt").write_text(text, encoding="utf-8")
    (reports_dir / "collective2_public_roster_report.html").write_text(
        "<html><body><pre>" + _html_escape(text) + "</pre></body></html>",
        encoding="utf-8",
    )
    write_json(reports_dir / "collective2_public_roster_report.json", {
        "run_utc": started.isoformat(),
        "access_summary": access_summary,
        "audit": audit,
        "ranked_systems": ranked_payload,
    })


def _candidate_report(candidate: SignalCandidate) -> dict[str, Any]:
    return {
        "system_id": candidate.signal.system_id,
        "system_name": candidate.system.system_name,
        "owner_screenname": candidate.system.owner_screenname,
        "signal_id": candidate.signal.signal_id,
        "ticker": candidate.signal.symbol,
        "action": candidate.signal.action,
        "quantity": candidate.signal.quantity,
        "latest_price_usd": candidate.latest_price_usd,
        "deterministic_score": candidate.deterministic_score,
        "system_score": candidate.system_score,
        "source_trade_notional_usd": candidate.source_trade_notional_usd,
        "source_trade_pct_of_portfolio": candidate.source_trade_pct_of_portfolio,
        "source_performance": asdict(candidate.performance),
        "consensus_count": candidate.consensus_count,
        "consensus_sources": candidate.consensus_sources,
        "reason": candidate.reason,
    }


def _render_email_html(
    *,
    state: CopyPortfolioState,
    equity: float,
    cfg: Collective2CopyConfig,
    started: datetime,
    events: list[dict[str, Any]],
    decisions: list[SignalDecision],
    candidates: list[SignalCandidate],
    access_summary: dict[str, Any],
) -> str:
    open_rows = "".join(
        "<tr>"
        f"<td>{_html_escape(p.ticker)}</td>"
        f"<td>{p.shares:.4f}</td>"
        f"<td>${p.entry_price_usd:,.2f}</td>"
        f"<td>{_html_escape(p.system_name)}</td>"
        "</tr>"
        for p in state.positions
    ) or "<tr><td colspan='4'>No open positions.</td></tr>"
    event_rows = "".join(_event_html_row(e) for e in events) or "<tr><td colspan='7'>No new paper trades this run.</td></tr>"
    decision_rows = "".join(
        "<tr>"
        f"<td>{_html_escape(d.decision)}</td>"
        f"<td>{_html_escape(d.ticker)}</td>"
        f"<td>{d.target_weight:.1%}</td>"
        f"<td>{d.confidence:.2f}</td>"
        f"<td>{d.consensus_count}</td>"
        f"<td>{_html_escape(d.source_system_name)}</td>"
        f"<td>{_html_escape(d.explanation or d.reason)}</td>"
        "</tr>"
        for d in decisions[:25]
    ) or "<tr><td colspan='7'>No decisions this run.</td></tr>"
    source_rows = "".join(_candidate_html_row(c) for c in candidates[:25]) or "<tr><td colspan='8'>No candidate signals.</td></tr>"
    return f"""<!doctype html>
<html>
<body style="font-family:Arial,sans-serif;color:#1f2933;">
  <h2>Collective2 Copy-Trade Paper Portfolio</h2>
  <p><strong>Run UTC:</strong> {_html_escape(started.isoformat())} | <strong>Paper only:</strong> yes</p>
  <h3>Portfolio State</h3>
  <p><strong>Equity:</strong> ${equity:,.2f} &nbsp; <strong>Cash:</strong> ${state.cash_usd:,.2f} &nbsp; <strong>Realized P/L:</strong> ${state.realized_pnl_usd:,.2f} &nbsp; <strong>Open Positions:</strong> {len(state.positions)} / {cfg.max_positions}</p>
  <table border="1" cellpadding="6" cellspacing="0"><thead><tr><th>Ticker</th><th>Shares</th><th>Entry</th><th>Source Portfolio</th></tr></thead><tbody>{open_rows}</tbody></table>
  <h3>New Trades</h3>
  <table border="1" cellpadding="6" cellspacing="0"><thead><tr><th>Action</th><th>Ticker</th><th>Notional</th><th>Source Portfolio</th><th>Source Performance</th><th>Consensus</th><th>LLM Explanation</th></tr></thead><tbody>{event_rows}</tbody></table>
  <h3>Candidate Sources and Historical Performance</h3>
  <p>Access: roster={access_summary.get("roster_count")} eligible={access_summary.get("eligible_count")} accessible={access_summary.get("accessible_count")} ranked_accessible={access_summary.get("accessible_ranked_count")}</p>
  <table border="1" cellpadding="6" cellspacing="0"><thead><tr><th>Signal</th><th>Ticker</th><th>Source Portfolio</th><th>Win Rate</th><th>Total P/L</th><th>Source Trade %</th><th>Consensus</th><th>Score</th></tr></thead><tbody>{source_rows}</tbody></table>
  <h3>LLM Decisions</h3>
  <table border="1" cellpadding="6" cellspacing="0"><thead><tr><th>Decision</th><th>Ticker</th><th>Target</th><th>Confidence</th><th>Consensus</th><th>Source</th><th>Explanation</th></tr></thead><tbody>{decision_rows}</tbody></table>
</body>
</html>
"""


def _event_html_row(event: dict[str, Any]) -> str:
    perf = event.get("source_performance") if isinstance(event.get("source_performance"), dict) else {}
    win_rate = _float(perf.get("win_rate"), 0.0)
    total_pl = _float(perf.get("total_pl"), 0.0)
    source_pct = event.get("source_trade_pct_of_portfolio")
    source_pct_text = "unknown" if not isinstance(source_pct, (int, float)) else f"{float(source_pct):.1%}"
    source = event.get("system_name") or event.get("system_id") or ""
    consensus = int(_float(event.get("consensus_count"), 1))
    explanation = str(event.get("explanation") or event.get("reason") or "")
    return (
        "<tr>"
        f"<td>{_html_escape(str(event.get('action') or ''))}</td>"
        f"<td>{_html_escape(str(event.get('ticker') or ''))}</td>"
        f"<td>${_float(event.get('notional_usd'), 0.0):,.2f}</td>"
        f"<td>{_html_escape(str(source))}<br>ID: {_html_escape(str(event.get('system_id') or ''))}</td>"
        f"<td>Win {win_rate:.1%}<br>Total P/L ${total_pl:,.2f}<br>Source trade {source_pct_text}</td>"
        f"<td>{consensus} source(s)</td>"
        f"<td>{_html_escape(explanation)}</td>"
        "</tr>"
    )


def _candidate_html_row(candidate: SignalCandidate) -> str:
    source_pct = "unknown" if candidate.source_trade_pct_of_portfolio is None else f"{candidate.source_trade_pct_of_portfolio:.1%}"
    return (
        "<tr>"
        f"<td>{_html_escape(candidate.signal.action)}</td>"
        f"<td>{_html_escape(candidate.signal.symbol)}</td>"
        f"<td>{_html_escape(candidate.system.system_name)}<br>ID: {_html_escape(candidate.system.system_id)}</td>"
        f"<td>{candidate.performance.win_rate:.1%} ({candidate.performance.winning_trade_count}/{candidate.performance.closed_trade_count})</td>"
        f"<td>${candidate.performance.total_pl:,.2f}</td>"
        f"<td>{source_pct}</td>"
        f"<td>{candidate.consensus_count}</td>"
        f"<td>{candidate.deterministic_score:.2f}</td>"
        "</tr>"
    )


def _write_no_key_reports(cfg: Collective2CopyConfig, reports_dir: Path, logger) -> None:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "run_utc": now.isoformat(),
        "paper_only": True,
        "status": "missing_collective2_api_key",
        "message": "Set COLLECTIVE2_API_KEY in the environment or GitHub Actions secrets.",
    }
    write_json(reports_dir / "collective2_copy_trades.json", payload)
    write_json(reports_dir / "collective2_copy_portfolio.json", {
        "run_utc": now.isoformat(),
        "cash_usd": cfg.initial_cash_usd,
        "equity_usd": cfg.initial_cash_usd,
        "positions": [],
    })
    text = "COLLECTIVE2 COPY-TRADE PAPER PORTFOLIO (USD)\nNo COLLECTIVE2_API_KEY configured.\n"
    (reports_dir / "collective2_copy_report.txt").write_text(text, encoding="utf-8")
    (reports_dir / "collective2_copy_email.html").write_text(f"<html><body><pre>{text}</pre></body></html>", encoding="utf-8")
    logger.warning("COLLECTIVE2_API_KEY not configured; wrote no-key reports")


def _write_email_gate(reports_dir: Path, now_utc: datetime) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    should_email = _should_send_email(now_utc)
    (reports_dir / "collective2_should_email.txt").write_text("true\n" if should_email else "false\n", encoding="utf-8")


def _should_send_email(now_utc: datetime) -> bool:
    if os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch":
        return True
    ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    if ny.weekday() >= 5:
        return False
    open_time = ny.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= ny <= close_time


def _ny_time_window(minutes: int) -> tuple[str, str]:
    tz = ZoneInfo("America/New_York")
    end = datetime.now(tz=tz)
    start = end - timedelta(minutes=max(1, int(minutes)))
    fmt = "%Y-%m-%d %H:%M:%S"
    return start.strftime(fmt), end.strftime(fmt)


def _load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = read_json(path)
    except Exception:
        return set()
    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict) and isinstance(data.get("signal_ids"), list):
        return {str(x) for x in data["signal_ids"]}
    return set()


def _save_seen(path: Path, seen: set[str]) -> None:
    write_json(path, {"signal_ids": sorted(seen)[-5000:]})


def _system_exposure(state: CopyPortfolioState, latest_prices: dict[str, float], system_id: str) -> float:
    total = 0.0
    for p in state.positions:
        if p.system_id == system_id:
            total += float(p.shares) * float(latest_prices.get(p.ticker, p.entry_price_usd))
    return total


def _llm_prompt(candidates: list[SignalCandidate], state: CopyPortfolioState, latest_prices: dict[str, float]) -> str:
    payload = {
        "task": "Select the best paper-only Collective2 copy-trade decisions. Return JSON array only.",
        "constraints": {
            "allowed_decisions": ["ACCEPT", "REJECT", "EXIT"],
            "target_weight_range": [0.0, 0.20],
            "long_stocks_etfs_only": True,
            "paper_only": True,
        },
        "portfolio": {
            "cash_usd": state.cash_usd,
            "equity_usd": compute_equity_usd(state, latest_prices),
            "realized_pnl_usd": state.realized_pnl_usd,
            "positions": [asdict(p) for p in state.positions],
        },
        "candidates": [
            {
                "system_id": c.signal.system_id,
                "system_name": c.system.system_name,
                "owner": c.system.owner_screenname,
                "signal_id": c.signal.signal_id,
                "ticker": c.signal.symbol,
                "action": c.signal.action,
                "posted_time_utc": signal_posted_at(c.signal).isoformat() if signal_posted_at(c.signal) else c.signal.posted_time,
                "latest_price_usd": c.latest_price_usd,
                "deterministic_score": c.deterministic_score,
                "system_score": c.system_score,
                "source_trade_quantity": c.signal.quantity,
                "source_trade_notional_usd": c.source_trade_notional_usd,
                "source_trade_pct_of_portfolio": c.source_trade_pct_of_portfolio,
                "same_ticker_profitable_source_count": c.consensus_count,
                "same_ticker_profitable_sources": c.consensus_sources,
                "source_portfolio_performance": asdict(c.performance),
                "recent_trades": c.recent_trades[:8],
            }
            for c in candidates
        ],
        "required_output_schema": [
            {
                "system_id": "string",
                "signal_id": "string",
                "ticker": "string",
                "decision": "ACCEPT|REJECT|EXIT",
                "target_weight": 0.0,
                "confidence": 0.0,
                "reason": "short reason",
                "explanation": "specific explanation using source performance, source trade size, consensus, and portfolio fit",
                "risk_flags": ["optional short risks"],
            }
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _parse_llm_decisions(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    text = raw.strip()
    if "```" in text:
        text = text.replace("```json", "```")
        parts = text.split("```")
        text = max(parts, key=len).strip()
    try:
        data = json.loads(text)
    except Exception:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(text[start:end + 1])
        except Exception:
            return []
    if isinstance(data, dict):
        data = data.get("decisions", [])
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def _generate_mock_trades(system_id: str) -> list[C2Trade]:
    """Generate fake trades for a system for testing when access is denied."""
    tz = ZoneInfo("America/New_York")
    now_ny = datetime.now(tz=tz)
    # Use some blue-chip tickers for mock trades
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    idx = int(hashlib.md5(system_id.encode()).hexdigest(), 16) % len(tickers)
    
    trades = []
    for i in range(3):
        ticker = tickers[(idx + i) % len(tickers)]
        opened_at = now_ny - timedelta(minutes=10 * (i + 1))
        trades.append(C2Trade(
            system_id=system_id,
            trade_id=f"mock_{system_id}_{ticker}_{i}",
            symbol=ticker,
            instrument="stock",
            quantity=100.0,
            long_or_short="long",
            open_or_closed="open",
            opened_when=opened_at.strftime("%Y-%m-%d %H:%M:%S"),
            closed_when="",
            opening_price=150.0 + i,
            closing_price=0.0,
            pl=0.0,
            raw={},
        ))
    return trades


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

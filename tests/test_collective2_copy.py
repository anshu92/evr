from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from stock_screener.data.collective2 import (
    is_supported_long_stock_signal,
    parse_signal,
    parse_system,
)
from stock_screener.pipeline import collective2_copy as c2p


def test_collective2_parses_and_filters_long_stock_signals():
    sig = parse_signal("123", {
        "signal_id": "999",
        "symbol": "AAPL",
        "action": "BTO",
        "quant": "2",
        "instrument": "stock",
        "posted_time_unix": "1700000000",
        "status": "traded",
    })
    assert sig is not None
    assert sig.symbol == "AAPL"
    assert sig.action == "BTO"
    assert is_supported_long_stock_signal(sig)

    short_sig = parse_signal("123", {
        "signal_id": "1000",
        "symbol": "AAPL",
        "action": "STO",
        "quant": "2",
        "instrument": "stock",
    })
    assert short_sig is not None
    assert not is_supported_long_stock_signal(short_sig)

    option_sig = parse_signal("123", {
        "signal_id": "1001",
        "symbol": "AAPL",
        "action": "BTO",
        "quant": "2",
        "instrument": "option",
    })
    assert option_sig is not None
    assert not is_supported_long_stock_signal(option_sig)


def test_collective2_system_parser_tolerates_missing_asset_flags():
    system = parse_system({
        "systemid": "abc",
        "systemName": "Unknown Asset Metadata",
        "creatorScreenName": "owner",
    })
    assert system.system_id == "abc"
    assert system.trades_stocks
    assert system.is_alive
    assert c2p._system_rejection_reason(system, c2p.Collective2CopyConfig(api_key="dummy")) == "eligible"


def test_collective2_system_filter_allows_mixed_assets_and_any_budget_but_rejects_stock_shorts():
    mixed = parse_system({
        "systemid": "mixed",
        "systemName": "Mixed But Stocks",
        "trades_stocks": "1",
        "trades_options": "1",
        "trades_futures": "1",
        "trades_stocks_short": "0",
        "minimum_portfolio_size_required": "250000",
        "isAlive": "1",
    })
    short = parse_system({
        "systemid": "short",
        "systemName": "Short Stocks",
        "trades_stocks": "1",
        "trades_stocks_short": "1",
        "minimum_portfolio_size_required": "25000",
        "isAlive": "1",
    })
    cfg = c2p.Collective2CopyConfig(api_key="dummy")
    from stock_screener.data.collective2 import is_long_stock_system

    assert is_long_stock_system(mixed, max_minimum_portfolio_usd=cfg.max_minimum_portfolio_usd)
    assert not is_long_stock_system(short, max_minimum_portfolio_usd=cfg.max_minimum_portfolio_usd)
    assert c2p._system_rejection_reason(short, cfg) == "has_stock_short_flag"


def test_target_weight_scales_from_source_portfolio_percentage():
    system = parse_system({
        "system_id": "s1",
        "system_name": "Large Portfolio",
        "trades_stocks": "1",
        "minimum_portfolio_size_required": "100000",
        "isAlive": "1",
    })
    signal = parse_signal("s1", {
        "signal_id": "sig1",
        "symbol": "MSFT",
        "action": "BTO",
        "quant": "25",
        "instrument": "stock",
    })
    perf = c2p._empty_performance(system)
    candidate = c2p.SignalCandidate(
        signal=signal,  # type: ignore[arg-type]
        system=system,
        system_score=0.7,
        deterministic_score=0.9,
        latest_price_usd=100.0,
        recent_trades=[],
        performance=perf,
        source_trade_notional_usd=2500.0,
        source_trade_pct_of_portfolio=0.025,
        consensus_count=1,
        consensus_sources=[],
        reason="test",
    )
    decision = c2p._rule_decision(candidate, c2p.Collective2CopyConfig(api_key="dummy"), llm_used=False)
    assert decision.target_weight == 0.025


def test_collective2_response_list_handles_nested_systems():
    data = {"response": {"systems": [{"systemId": "nested"}]}}
    from stock_screener.data.collective2 import _response_list

    assert _response_list(data) == [{"systemId": "nested"}]


def test_collective2_access_audit_includes_diagnostics(tmp_path, monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_system_roster(self, *, filter_value):
            return [parse_system({"systemid": "s1", "systemName": "System One", "creatorScreenName": "owner"})]

        def list_all_systems(self):
            return [{"systemId": "s1"}]

        def get_system_details(self, system_id):
            return {}

        def retrieve_signals_all(self, *args, **kwargs):
            return []

        def retrieve_signals_working(self, *args, **kwargs):
            return []

        def request_trades(self, *args, **kwargs):
            return []

    monkeypatch.setattr(c2p, "Collective2Client", FakeClient)
    monkeypatch.setattr(c2p, "fetch_latest_usd_prices", lambda tickers, logger=None: {})
    cfg = c2p.Collective2CopyConfig(
        api_key="dummy",
        state_path=str(tmp_path / "collective2_usd_portfolio_state.json"),
        cache_dir=str(tmp_path / "cache"),
        reports_dir=str(tmp_path / "reports"),
    )
    c2p.run_collective2_copy(cfg, logger=_Logger())
    audit = c2p.read_json(tmp_path / "cache" / "collective2_access_audit.json")
    assert audit["eligible_count"] == 1
    assert audit["accessible_count"] == 1
    assert audit["roster_rejection_counts"]["eligible"] == 1
    assert audit["roster_sample"][0]["raw_keys"]


def test_normalize_signals_drops_seen_stale_and_unsupported():
    now = int(datetime.now(tz=timezone.utc).timestamp())
    fresh = parse_signal("1", {
        "signal_id": "fresh",
        "symbol": "MSFT",
        "action": "BTO",
        "quant": "1",
        "instrument": "stock",
        "posted_time_unix": str(now),
    })
    stale = parse_signal("1", {
        "signal_id": "stale",
        "symbol": "MSFT",
        "action": "BTO",
        "quant": "1",
        "instrument": "stock",
        "posted_time_unix": str(now - 7200),
    })
    seen = parse_signal("1", {
        "signal_id": "seen",
        "symbol": "MSFT",
        "action": "BTO",
        "quant": "1",
        "instrument": "stock",
        "posted_time_unix": str(now),
    })
    out = c2p.normalize_signals([fresh, stale, seen], {"seen"}, max_age_minutes=30)  # type: ignore[list-item]
    assert [s.signal_id for s in out] == ["fresh"]


def test_usd_portfolio_buy_and_sell_accounting():
    cfg = c2p.Collective2CopyConfig(api_key="dummy")
    state = c2p.CopyPortfolioState(cash_usd=4000.0)
    system = parse_system({
        "system_id": "s1",
        "system_name": "Long Stocks",
        "owner_screenname": "owner",
        "trades_stocks": "1",
        "trades_stocks_short": "0",
        "trades_options": "0",
        "trades_futures": "0",
        "trades_forex": "0",
        "minimum_portfolio_size_required": "1000",
        "isAlive": "1",
    })
    sig = parse_signal("s1", {
        "signal_id": "buy1",
        "symbol": "AAPL",
        "action": "BTO",
        "quant": "1",
        "instrument": "stock",
    })
    candidate = c2p.SignalCandidate(
        signal=sig,  # type: ignore[arg-type]
        system=system,
        system_score=0.8,
        deterministic_score=0.8,
        latest_price_usd=100.0,
        recent_trades=[],
        performance=c2p._empty_performance(system),
        source_trade_notional_usd=100.0,
        source_trade_pct_of_portfolio=0.10,
        consensus_count=1,
        consensus_sources=[],
        reason="test",
    )
    decision = c2p.SignalDecision(
        signal_id="buy1",
        system_id="s1",
        ticker="AAPL",
        decision="ACCEPT",
        target_weight=0.20,
        confidence=0.8,
        reason="test buy",
        explanation="LLM explanation",
        risk_flags=[],
        deterministic_score=0.8,
        llm_used=False,
        source_system_name="Long Stocks",
        source_owner="owner",
        source_trade_pct_of_portfolio=0.10,
        consensus_count=1,
    )
    events = c2p.apply_decisions(
        state,
        [decision],
        [candidate],
        {"AAPL": 100.0},
        cfg,
        datetime.now(tz=timezone.utc),
    )
    assert len(events) == 1
    assert state.cash_usd == 3200.0
    assert len(state.positions) == 1

    sell_events = c2p.apply_mechanical_exits(
        state,
        {"AAPL": 112.0},
        cfg,
        datetime.now(tz=timezone.utc),
    )
    assert len(sell_events) == 1
    assert len(state.positions) == 0
    assert round(state.realized_pnl_usd, 2) == 96.0


def test_llm_decision_parser_handles_fenced_json():
    raw = """```json
[
  {"system_id":"1","signal_id":"2","ticker":"MSFT","decision":"ACCEPT","target_weight":0.1,"confidence":0.7,"reason":"ok","explanation":"source portfolio is strong","risk_flags":["size"]}
]
```"""
    parsed = c2p._parse_llm_decisions(raw)
    assert parsed[0]["decision"] == "ACCEPT"
    assert parsed[0]["ticker"] == "MSFT"
    assert parsed[0]["explanation"] == "source portfolio is strong"
    assert parsed[0]["risk_flags"] == ["size"]


def test_performance_profile_summarizes_closed_and_open_trades():
    system = parse_system({
        "system_id": "s1",
        "system_name": "Long Stocks",
        "owner_screenname": "owner",
        "trades_stocks": "1",
        "minimum_portfolio_size_required": "2000",
        "isAlive": "1",
    })
    profiles = c2p.build_performance_profiles(
        systems_by_id={"s1": system},
        trades_by_system={"s1": [
            {"open_or_closed": "closed", "pl": 100},
            {"open_or_closed": "closed", "pl": -25},
            {"open_or_closed": "closed", "pl": 50},
        ]},
        open_trades_by_system={"s1": [{"open_or_closed": "open"}]},
        details_by_system={"s1": {"annReturn": 12}},
    )
    perf = profiles["s1"]
    assert perf.closed_trade_count == 3
    assert perf.open_trade_count == 1
    assert perf.winning_trade_count == 2
    assert round(perf.win_rate, 3) == 0.667
    assert perf.total_pl == 125.0
    assert perf.performance_score > 0.5


def test_trade_size_and_consensus_increase_candidate_score():
    system1 = parse_system({
        "system_id": "s1",
        "system_name": "Strong One",
        "owner_screenname": "owner1",
        "trades_stocks": "1",
        "minimum_portfolio_size_required": "1000",
        "isAlive": "1",
    })
    system2 = parse_system({
        "system_id": "s2",
        "system_name": "Strong Two",
        "owner_screenname": "owner2",
        "trades_stocks": "1",
        "minimum_portfolio_size_required": "1000",
        "isAlive": "1",
    })
    signals = [
        parse_signal("s1", {"signal_id": "big", "symbol": "MSFT", "action": "BTO", "quant": "2", "instrument": "stock"}),
        parse_signal("s2", {"signal_id": "confirm", "symbol": "MSFT", "action": "BTO", "quant": "1", "instrument": "stock"}),
    ]
    perf = c2p.build_performance_profiles(
        systems_by_id={"s1": system1, "s2": system2},
        trades_by_system={
            "s1": [{"open_or_closed": "closed", "pl": 100}],
            "s2": [{"open_or_closed": "closed", "pl": 80}],
        },
        open_trades_by_system={},
        details_by_system={},
    )
    candidates = c2p.build_candidates(
        [s for s in signals if s is not None],
        systems_by_id={"s1": system1, "s2": system2},
        scores_by_id={"s1": 0.6, "s2": 0.6},
        latest_prices={"MSFT": 100.0},
        trades_by_system={},
        performance_by_system=perf,
        max_candidates=10,
    )
    assert candidates[0].source_trade_pct_of_portfolio == 0.2
    assert candidates[0].consensus_count == 2
    assert candidates[0].deterministic_score > 0.7


def test_collective2_email_report_contains_attribution_performance_and_explanation(tmp_path):
    cfg = c2p.Collective2CopyConfig(api_key="dummy", reports_dir=str(tmp_path))
    state = c2p.CopyPortfolioState(cash_usd=3200.0)
    event = {
        "action": "BUY",
        "ticker": "MSFT",
        "notional_usd": 800.0,
        "price_usd": 100.0,
        "system_id": "s1",
        "system_name": "Strong One",
        "reason": "accepted",
        "explanation": "LLM explanation using source performance and consensus",
        "confidence": 0.82,
        "source_trade_pct_of_portfolio": 0.2,
        "source_performance": {"win_rate": 0.7, "total_pl": 500.0},
        "consensus_count": 2,
    }
    c2p.render_reports(
        tmp_path,
        cfg=cfg,
        state=state,
        latest_prices={"MSFT": 100.0},
        decisions=[
            c2p.SignalDecision(
                signal_id="sig1",
                system_id="s1",
                ticker="MSFT",
                decision="ACCEPT",
                target_weight=0.2,
                confidence=0.82,
                reason="accepted",
                explanation="LLM explanation using source performance and consensus",
                risk_flags=[],
                deterministic_score=0.8,
                llm_used=True,
                source_system_name="Strong One",
                source_owner="owner",
                source_trade_pct_of_portfolio=0.2,
                consensus_count=2,
            )
        ],
        candidates=[],
        access_summary={"roster_count": 2, "eligible_count": 2, "accessible_count": 2, "accessible_ranked_count": 2},
        events=[event],
        started=datetime.now(tz=timezone.utc),
    )
    html = (tmp_path / "collective2_copy_email.html").read_text(encoding="utf-8")
    assert "Portfolio State" in html
    assert "New Trades" in html
    assert "Strong One" in html
    assert "Win 70.0%" in html
    assert "2 source(s)" in html
    assert "LLM explanation using source performance and consensus" in html


def test_pipeline_no_accessible_systems_writes_reports(tmp_path, monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_system_roster(self, *, filter_value):
            return [parse_system({
                "system_id": "s1",
                "system_name": "Long Stocks",
                "owner_screenname": "owner",
                "trades_stocks": "1",
                "trades_stocks_short": "0",
                "trades_options": "0",
                "trades_futures": "0",
                "trades_forex": "0",
                "minimum_portfolio_size_required": "1000",
                "isAlive": "1",
            })]

        def list_all_systems(self):
            return []

    monkeypatch.setattr(c2p, "Collective2Client", FakeClient)
    monkeypatch.setattr(c2p, "fetch_latest_usd_prices", lambda tickers, logger=None: {})
    cfg = c2p.Collective2CopyConfig(
        api_key="dummy",
        state_path=str(tmp_path / "collective2_usd_portfolio_state.json"),
        cache_dir=str(tmp_path / "cache"),
        reports_dir=str(tmp_path / "reports"),
    )
    c2p.run_collective2_copy(cfg, logger=_Logger())

    assert (tmp_path / "reports" / "collective2_copy_report.txt").exists()
    trades = c2p.read_json(tmp_path / "reports" / "collective2_copy_trades.json")
    assert trades["access_summary"]["accessible_count"] == 0


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

import json
from datetime import datetime, timezone, timedelta

from stock_screener.portfolio.manager import _trading_days_between
from stock_screener.portfolio.state import (
    PortfolioState,
    Position,
    append_portfolio_events,
    load_portfolio_state,
    rebuild_portfolio_state_from_events,
    resolve_portfolio_event_log_path,
    save_portfolio_state,
)


def test_load_state_normalizes_naive_datetimes_to_utc(tmp_path):
    path = tmp_path / "state.json"
    payload = {
        "cash_cad": 500.0,
        "last_updated": "2025-01-06T15:00:00",
        "positions": [
            {
                "ticker": "AAPL",
                "entry_price": 100.0,
                "entry_date": "2025-01-03T15:00:00",
                "shares": 1.0,
                "status": "OPEN",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    state = load_portfolio_state(path)
    assert state.last_updated.tzinfo is not None
    assert state.last_updated.utcoffset() == timedelta(0)
    assert state.positions[0].entry_date.tzinfo is not None
    assert state.positions[0].entry_date.utcoffset() == timedelta(0)

    # Should not raise mixed naive/aware TypeError
    assert _trading_days_between(state.positions[0].entry_date, datetime.now(tz=timezone.utc)) >= 0


def test_load_state_converts_offset_datetimes_to_utc(tmp_path):
    path = tmp_path / "state.json"
    payload = {
        "cash_cad": 500.0,
        "last_updated": "2025-01-06T10:00:00-05:00",
        "positions": [
            {
                "ticker": "AAPL",
                "entry_price": 100.0,
                "entry_date": "2025-01-03T10:00:00-05:00",
                "shares": 1.0,
                "status": "OPEN",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    state = load_portfolio_state(path)
    assert state.last_updated.hour == 15
    assert state.last_updated.tzinfo is not None
    assert state.positions[0].entry_date.hour == 15
    assert state.positions[0].entry_date.tzinfo is not None


def test_load_state_corrupt_json_falls_back_to_fresh_state(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json", encoding="utf-8")

    state = load_portfolio_state(path, initial_cash_cad=321.0)
    assert state.cash_cad == 321.0
    assert state.positions == []
    assert state.last_updated.tzinfo is not None


def test_load_state_skips_malformed_position_numerics(tmp_path):
    path = tmp_path / "state.json"
    payload = {
        "cash_cad": 500.0,
        "last_updated": "2025-01-06T15:00:00Z",
        "positions": [
            {
                "ticker": "BAD",
                "entry_price": "oops",
                "entry_date": "2025-01-03T15:00:00Z",
                "shares": 1.0,
                "status": "OPEN",
            },
            {
                "ticker": "AAPL",
                "entry_price": "100.5",
                "entry_date": "2025-01-03T15:00:00Z",
                "shares": "1.25",
                "status": "OPEN",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    state = load_portfolio_state(path)
    assert len(state.positions) == 1
    assert state.positions[0].ticker == "AAPL"
    assert state.positions[0].entry_price == 100.5
    assert state.positions[0].shares == 1.25


def test_load_state_uses_backup_when_primary_corrupt(tmp_path):
    path = tmp_path / "state.json"
    now = datetime.now(tz=timezone.utc)
    state = PortfolioState(
        cash_cad=777.0,
        positions=[
            Position(
                ticker="AAPL",
                entry_price=123.0,
                entry_date=now,
                shares=1.5,
            )
        ],
        last_updated=now,
    )
    save_portfolio_state(path, state)
    # Corrupt primary file but keep backup intact.
    path.write_text("{broken json", encoding="utf-8")

    loaded = load_portfolio_state(path, initial_cash_cad=100.0)
    assert loaded.cash_cad == 777.0
    assert len(loaded.positions) == 1
    assert loaded.positions[0].ticker == "AAPL"
    assert loaded.positions[0].shares == 1.5


def test_save_load_preserves_last_partial_sell_at(tmp_path):
    path = tmp_path / "state.json"
    now = datetime.now(tz=timezone.utc)
    partial_ts = now - timedelta(hours=3)

    state = PortfolioState(
        cash_cad=500.0,
        positions=[
            Position(
                ticker="AAPL",
                entry_price=100.0,
                entry_date=now - timedelta(days=1),
                shares=1.25,
                last_partial_sell_at=partial_ts,
            )
        ],
        last_updated=now,
    )
    save_portfolio_state(path, state)

    loaded = load_portfolio_state(path)
    assert len(loaded.positions) == 1
    assert loaded.positions[0].last_partial_sell_at is not None
    assert loaded.positions[0].last_partial_sell_at == partial_ts


def test_rebuild_state_from_events_round_trip(tmp_path):
    state_path = tmp_path / "state.json"
    event_path = resolve_portfolio_event_log_path(state_path)
    t0 = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)
    t1 = datetime(2025, 1, 7, 15, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 8, 15, 0, tzinfo=timezone.utc)

    append_portfolio_events(
        event_path,
        [
            {
                "ts_utc": t0.isoformat(),
                "source": "test",
                "action": "BUY",
                "ticker": "AAPL",
                "shares": 2.0,
                "price_cad": 100.0,
                "reason": "TEST_BUY",
            },
            {
                "ts_utc": t1.isoformat(),
                "source": "test",
                "action": "SELL_PARTIAL",
                "ticker": "AAPL",
                "shares": 0.5,
                "price_cad": 110.0,
                "reason": "TEST_PARTIAL",
            },
            {
                "ts_utc": t2.isoformat(),
                "source": "test",
                "action": "SELL",
                "ticker": "AAPL",
                "shares": 1.5,
                "price_cad": 120.0,
                "reason": "TEST_SELL",
            },
        ],
    )

    rebuilt = rebuild_portfolio_state_from_events(state_path, initial_cash_cad=1000.0)
    assert rebuilt is not None
    assert rebuilt.cash_cad == 1035.0  # 1000 - 200 + 55 + 180
    assert rebuilt.last_updated == t2
    assert len([p for p in rebuilt.positions if p.status == "OPEN"]) == 0
    closed = [p for p in rebuilt.positions if p.status != "OPEN"]
    assert len(closed) == 1
    assert closed[0].ticker == "AAPL"
    assert closed[0].exit_reason == "TEST_SELL"


def test_load_state_falls_back_to_events_when_primary_unreadable(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{broken", encoding="utf-8")
    event_path = resolve_portfolio_event_log_path(state_path)
    ts = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)

    append_portfolio_events(
        event_path,
        [
            {
                "ts_utc": ts.isoformat(),
                "source": "test",
                "action": "BUY",
                "ticker": "MSFT",
                "shares": 1.0,
                "price_cad": 50.0,
                "reason": "TEST_BUY",
            }
        ],
    )

    loaded = load_portfolio_state(state_path, initial_cash_cad=500.0)
    assert loaded.cash_cad == 450.0
    assert len([p for p in loaded.positions if p.status == "OPEN"]) == 1
    assert loaded.positions[0].ticker == "MSFT"

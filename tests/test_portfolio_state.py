import json
from datetime import datetime, timezone, timedelta

from stock_screener.portfolio.manager import _trading_days_between
from stock_screener.portfolio.state import load_portfolio_state, save_portfolio_state, PortfolioState, Position


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

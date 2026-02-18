import json
from datetime import datetime, timezone, timedelta

from stock_screener.portfolio.manager import _trading_days_between
from stock_screener.portfolio.state import load_portfolio_state


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

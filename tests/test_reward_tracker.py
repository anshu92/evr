"""Tests for the reward tracker (RewardEntry, RewardLog)."""
import json

import pandas as pd
import pytest

from stock_screener.reward.tracker import RewardEntry, RewardLog


def _make_entry(date: str = "2026-01-15", ticker: str = "AAPL", pred: float = 0.05, **kw) -> RewardEntry:
    return RewardEntry(date=date, ticker=ticker, predicted_return=pred, **kw)


class TestRewardEntry:
    def test_key(self):
        e = _make_entry()
        assert e.key() == ("2026-01-15", "AAPL", "PREDICTION")

    def test_defaults(self):
        e = _make_entry()
        assert e.realized_1d_return is None
        assert e.exit_reason is None
        assert e.per_model_preds is None


class TestRewardLog:
    def test_empty(self):
        log = RewardLog()
        assert len(log.entries) == 0
        assert log.to_dataframe().empty

    def test_append_dedup(self):
        log = RewardLog()
        e1 = _make_entry(pred=0.01)
        e2 = _make_entry(pred=0.02)  # Same (date, ticker) -> replaces e1
        log.append(e1)
        log.append(e2)
        assert len(log.entries) == 1
        assert log.entries[0].predicted_return == 0.02

    def test_append_batch(self):
        log = RewardLog()
        entries = [
            _make_entry(ticker="AAPL"),
            _make_entry(ticker="MSFT"),
            _make_entry(ticker="GOOG"),
        ]
        log.append_batch(entries)
        assert len(log.entries) == 3

    def test_append_batch_dedup(self):
        log = RewardLog()
        log.append(_make_entry(ticker="AAPL", pred=0.01))
        log.append(_make_entry(ticker="MSFT", pred=0.02))

        # Batch replaces AAPL, keeps MSFT, adds GOOG
        batch = [
            _make_entry(ticker="AAPL", pred=0.10),
            _make_entry(ticker="GOOG", pred=0.03),
        ]
        log.append_batch(batch)
        assert len(log.entries) == 3
        aapl = [e for e in log.entries if e.ticker == "AAPL"][0]
        assert aapl.predicted_return == 0.10

    def test_prediction_and_close_entries_can_coexist(self):
        log = RewardLog()
        log.append_batch([
            _make_entry(date="2026-01-15", ticker="AAPL", pred=0.03, event_type="PREDICTION"),
            _make_entry(
                date="2026-01-15",
                ticker="AAPL",
                pred=0.03,
                event_type="CLOSE",
                exit_reason="TAKE_PROFIT",
                realized_cumulative_return=0.06,
            ),
        ])
        assert len(log.entries) == 2
        assert {e.event_type for e in log.entries} == {"PREDICTION", "CLOSE"}

    def test_save_load_roundtrip(self, tmp_path):
        log = RewardLog()
        log.append(_make_entry(ticker="AAPL", pred=0.05))
        log.append(_make_entry(ticker="MSFT", pred=-0.02, realized_1d_return=-0.01))

        path = tmp_path / "reward_log.json"
        log.save(path)
        loaded = RewardLog.load(path)
        assert len(loaded.entries) == 2
        msft = [e for e in loaded.entries if e.ticker == "MSFT"][0]
        assert msft.realized_1d_return == -0.01

    def test_load_missing_file(self, tmp_path):
        log = RewardLog.load(tmp_path / "nonexistent.json")
        assert len(log.entries) == 0

    def test_load_corrupt_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json!!!")
        log = RewardLog.load(path)
        assert len(log.entries) == 0

    def test_update_realized_returns(self):
        log = RewardLog()
        log.append(_make_entry(date="2026-01-14", ticker="AAPL", price_at_prediction=150.0))
        log.append(_make_entry(date="2026-01-14", ticker="MSFT", price_at_prediction=400.0))

        prices = pd.Series({"AAPL": 153.0, "MSFT": 396.0})
        updated = log.update_realized_returns(prices, date_str="2026-01-15")
        assert updated == 2
        aapl = [e for e in log.entries if e.ticker == "AAPL"][0]
        assert abs(aapl.realized_1d_return - 0.02) < 1e-6
        msft = [e for e in log.entries if e.ticker == "MSFT"][0]
        assert abs(msft.realized_1d_return - (-0.01)) < 1e-6

    def test_update_realized_returns_skips_close_events(self):
        log = RewardLog()
        log.append(_make_entry(date="2026-01-14", ticker="AAPL", price_at_prediction=150.0))
        log.append(_make_entry(
            date="2026-01-14",
            ticker="AAPL",
            price_at_prediction=150.0,
            event_type="CLOSE",
            exit_reason="TAKE_PROFIT",
        ))
        prices = pd.Series({"AAPL": 153.0})
        updated = log.update_realized_returns(prices, date_str="2026-01-15")
        assert updated == 1
        pred_row = [e for e in log.entries if e.event_type == "PREDICTION"][0]
        close_row = [e for e in log.entries if e.event_type == "CLOSE"][0]
        assert pred_row.realized_1d_return is not None
        assert close_row.realized_1d_return is None

    def test_update_skips_same_day(self):
        log = RewardLog()
        log.append(_make_entry(date="2026-01-15", ticker="AAPL", price_at_prediction=150.0))
        prices = pd.Series({"AAPL": 153.0})
        updated = log.update_realized_returns(prices, date_str="2026-01-15")
        assert updated == 0  # Same day -> skip

    def test_update_skips_already_filled(self):
        log = RewardLog()
        log.append(_make_entry(date="2026-01-14", ticker="AAPL", price_at_prediction=150.0, realized_1d_return=0.01))
        prices = pd.Series({"AAPL": 999.0})
        updated = log.update_realized_returns(prices, date_str="2026-01-15")
        assert updated == 0  # Already has a realized return

    def test_update_only_latest_unresolved_date(self):
        log = RewardLog()
        log.append(_make_entry(date="2026-01-10", ticker="OLD", price_at_prediction=100.0))
        log.append(_make_entry(date="2026-01-14", ticker="NEW", price_at_prediction=200.0))

        prices = pd.Series({"OLD": 110.0, "NEW": 210.0})
        updated = log.update_realized_returns(prices, date_str="2026-01-15")
        assert updated == 1

        old = [e for e in log.entries if e.ticker == "OLD"][0]
        new = [e for e in log.entries if e.ticker == "NEW"][0]
        assert old.realized_1d_return is None
        assert new.realized_1d_return is not None

    def test_closed_entries(self):
        log = RewardLog()
        log.append(_make_entry(ticker="AAPL"))
        log.append(_make_entry(ticker="MSFT", exit_reason="STOP_LOSS"))
        closed = log.closed_entries()
        assert len(closed) == 1
        assert closed[0].ticker == "MSFT"

    def test_to_dataframe(self):
        log = RewardLog()
        log.append(_make_entry(ticker="AAPL"))
        log.append(_make_entry(ticker="MSFT"))
        df = log.to_dataframe()
        assert len(df) == 2
        assert "ticker" in df.columns
        assert "predicted_return" in df.columns

    def test_entries_for_window(self):
        log = RewardLog()
        # Add an old entry and a recent one
        log.append(_make_entry(date="2020-01-01", ticker="OLD"))
        log.append(_make_entry(date="2026-02-04", ticker="NEW"))
        recent = log.entries_for_window(last_n_days=30)
        assert len(recent) == 1
        assert recent[0].ticker == "NEW"

    def test_trim_old_entries(self):
        log = RewardLog(max_days=30)
        log.append(_make_entry(date="2020-01-01", ticker="OLD"))
        log.append(_make_entry(date="2026-02-04", ticker="NEW"))
        # _trim is called on append
        assert len(log.entries) == 1
        assert log.entries[0].ticker == "NEW"

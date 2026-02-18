"""Tests for per-action reward tracking and composite scoring."""
import numpy as np
import pandas as pd
import pytest

from stock_screener.reward.tracker import ActionRewardEntry, ActionRewardLog
from stock_screener.reward.feedback import (
    ActionScoreParams,
    score_action,
    score_actions,
    compute_action_quality_summary,
    _best_post_return,
)


def _sell_entry(**kw) -> ActionRewardEntry:
    defaults = dict(
        date="2026-02-01", ticker="AAPL", action="SELL", reason="TAKE_PROFIT",
        price_at_action=110.0, shares=5.0, predicted_return=0.05,
        entry_price=100.0, confidence=0.7,
    )
    defaults.update(kw)
    return ActionRewardEntry(**defaults)


def _buy_entry(**kw) -> ActionRewardEntry:
    defaults = dict(
        date="2026-02-01", ticker="MSFT", action="BUY", reason="NEW_POSITION",
        price_at_action=400.0, shares=1.0, predicted_return=0.08, confidence=0.6,
    )
    defaults.update(kw)
    return ActionRewardEntry(**defaults)


def _hold_entry(**kw) -> ActionRewardEntry:
    defaults = dict(
        date="2026-02-01", ticker="GOOG", action="HOLD", reason="IN_TARGET",
        price_at_action=150.0, shares=3.0, predicted_return=0.03, confidence=0.5,
    )
    defaults.update(kw)
    return ActionRewardEntry(**defaults)


# ---------------------------------------------------------------------------
# ActionRewardEntry / ActionRewardLog basics
# ---------------------------------------------------------------------------

class TestActionRewardEntry:
    def test_key(self):
        e = _sell_entry()
        assert e.key() == ("2026-02-01", "AAPL", "SELL")

    def test_defaults_none(self):
        e = _buy_entry()
        assert e.price_1d_after is None
        assert e.return_5d is None
        assert e.action_reward is None
        assert e.reward_components is None

    def test_new_fields_default_none(self):
        e = _sell_entry()
        assert e.replaced_by_ticker is None
        assert e.replaced_by_price is None
        assert e.rotation_alpha is None
        assert e.screened_avg_pred_return is None
        assert e.stock_volatility is None
        assert e.vol_adjusted_return_5d is None
        assert e.confidence_calibration is None
        assert e.selection_alpha_5d is None


class TestActionRewardLog:
    def test_empty(self):
        log = ActionRewardLog()
        assert len(log.entries) == 0
        assert log.to_dataframe().empty

    def test_append_batch_dedup(self):
        log = ActionRewardLog()
        e1 = _sell_entry(predicted_return=0.01)
        e2 = _sell_entry(predicted_return=0.09)
        log.append_batch([e1])
        log.append_batch([e2])
        assert len(log.entries) == 1
        assert log.entries[0].predicted_return == 0.09

    def test_different_actions_same_ticker(self):
        log = ActionRewardLog()
        log.append_batch([
            _buy_entry(ticker="AAPL"),
            _sell_entry(ticker="AAPL"),
        ])
        assert len(log.entries) == 2

    def test_save_load_roundtrip(self, tmp_path):
        log = ActionRewardLog()
        e = _sell_entry(replaced_by_ticker="GOOG", replaced_by_price=150.0,
                        stock_volatility=0.3, screened_avg_pred_return=0.04)
        log.append_batch([e, _buy_entry(), _hold_entry()])
        path = tmp_path / "actions.json"
        log.save(path)
        loaded = ActionRewardLog.load(path)
        assert len(loaded.entries) == 3
        sell_loaded = [x for x in loaded.entries if x.action == "SELL"][0]
        assert sell_loaded.replaced_by_ticker == "GOOG"
        assert sell_loaded.replaced_by_price == 150.0
        assert sell_loaded.stock_volatility == 0.3
        assert sell_loaded.screened_avg_pred_return == 0.04

    def test_save_load_components(self, tmp_path):
        log = ActionRewardLog()
        e = _sell_entry()
        e.action_reward = 0.05
        e.reward_components = {"base_outcome": 0.10, "opportunity_cost": -0.02}
        log.append_batch([e])
        path = tmp_path / "actions.json"
        log.save(path)
        loaded = ActionRewardLog.load(path)
        assert loaded.entries[0].reward_components is not None
        assert loaded.entries[0].reward_components["base_outcome"] == 0.10

    def test_load_missing(self, tmp_path):
        log = ActionRewardLog.load(tmp_path / "nope.json")
        assert len(log.entries) == 0

    def test_scored_entries(self):
        log = ActionRewardLog()
        e1 = _sell_entry()
        e1.action_reward = 0.05
        e2 = _buy_entry()
        log.append_batch([e1, e2])
        assert len(log.scored_entries()) == 1

    def test_trim(self):
        log = ActionRewardLog(max_days=30)
        log.append_batch([
            _sell_entry(date="2020-01-01"),
            _sell_entry(date="2026-02-04", ticker="NEW"),
        ])
        assert len(log.entries) == 1
        assert log.entries[0].ticker == "NEW"


# ---------------------------------------------------------------------------
# Backfill prices
# ---------------------------------------------------------------------------

class TestBackfillPrices:
    def test_fills_1d(self):
        log = ActionRewardLog()
        log.append_batch([_sell_entry(date="2026-02-03", price_at_action=100.0)])
        prices = pd.Series({"AAPL": 105.0})
        n = log.backfill_prices(prices, "2026-02-04")
        assert n >= 1
        e = log.entries[0]
        assert e.price_1d_after == 105.0
        assert abs(e.return_1d - 0.05) < 1e-6

    def test_uses_trading_day_horizons(self):
        """Friday->Monday should count as one trading day, not three calendar days."""
        log = ActionRewardLog()
        log.append_batch([_sell_entry(date="2026-01-02", price_at_action=100.0)])
        prices = pd.Series({"AAPL": 101.0})
        log.backfill_prices(prices, "2026-01-05")
        e = log.entries[0]
        assert e.return_1d is not None
        assert e.return_3d is None

    def test_fills_5d_and_classifies_sell(self):
        log = ActionRewardLog()
        log.append_batch([_sell_entry(date="2026-01-27", price_at_action=100.0, entry_price=90.0)])
        prices = pd.Series({"AAPL": 105.0})
        log.backfill_prices(prices, "2026-02-03")
        e = log.entries[0]
        assert e.return_5d is not None
        assert abs(e.return_5d - 0.05) < 1e-6
        assert e.was_premature_sell is True

    def test_fills_5d_classifies_good_sell(self):
        log = ActionRewardLog()
        log.append_batch([_sell_entry(date="2026-01-27", price_at_action=100.0)])
        prices = pd.Series({"AAPL": 95.0})
        log.backfill_prices(prices, "2026-02-03")
        e = log.entries[0]
        assert e.was_premature_sell is False

    def test_fills_5d_classifies_bad_buy(self):
        log = ActionRewardLog()
        log.append_batch([_buy_entry(date="2026-01-27", price_at_action=400.0)])
        prices = pd.Series({"MSFT": 380.0})
        log.backfill_prices(prices, "2026-02-03")
        e = log.entries[0]
        assert e.was_bad_buy is True

    def test_fills_5d_classifies_wrong_hold(self):
        log = ActionRewardLog()
        log.append_batch([_hold_entry(date="2026-01-27", price_at_action=150.0)])
        prices = pd.Series({"GOOG": 140.0})
        log.backfill_prices(prices, "2026-02-03")
        e = log.entries[0]
        assert e.was_wrong_hold is True

    def test_skips_no_price(self):
        log = ActionRewardLog()
        log.append_batch([_sell_entry(date="2026-02-03")])
        prices = pd.Series({"MSFT": 200.0})
        n = log.backfill_prices(prices, "2026-02-04")
        assert n == 0

    def test_rotation_counterpart_sell(self):
        """Backfill computes rotation_alpha for SELL with replacement."""
        log = ActionRewardLog()
        e = _sell_entry(
            date="2026-01-27", price_at_action=100.0,
            replaced_by_ticker="GOOG", replaced_by_price=150.0,
        )
        log.append_batch([e])
        # AAPL went to 95 (sold correctly), GOOG went to 165 (+10%)
        prices = pd.Series({"AAPL": 95.0, "GOOG": 165.0})
        log.backfill_prices(prices, "2026-02-03")
        e = log.entries[0]
        assert e.replaced_by_return_5d is not None
        assert abs(e.replaced_by_return_5d - 0.10) < 1e-6
        # rotation_alpha = replacement_return - sold_stock_return = 0.10 - (-0.05)
        assert e.rotation_alpha is not None
        assert abs(e.rotation_alpha - 0.15) < 1e-6

    def test_rotation_counterpart_buy(self):
        """Backfill computes rotation_alpha for BUY replacing a sold stock."""
        log = ActionRewardLog()
        e = _buy_entry(
            date="2026-01-27", price_at_action=400.0,
            replaced_ticker="AAPL", replaced_price=100.0,
        )
        log.append_batch([e])
        # MSFT went to 420 (+5%), AAPL went to 110 (+10%)
        prices = pd.Series({"MSFT": 420.0, "AAPL": 110.0})
        log.backfill_prices(prices, "2026-02-03")
        e = log.entries[0]
        assert e.replaced_return_5d is not None
        assert abs(e.replaced_return_5d - 0.10) < 1e-6
        # rotation_alpha = our_return - replaced_return = 0.05 - 0.10 = -0.05
        assert e.rotation_alpha is not None
        assert abs(e.rotation_alpha - (-0.05)) < 1e-6

    def test_vol_adjusted_return(self):
        """Backfill computes vol-adjusted return when volatility is set."""
        log = ActionRewardLog()
        e = _buy_entry(date="2026-01-27", price_at_action=400.0, stock_volatility=0.30)
        log.append_batch([e])
        prices = pd.Series({"MSFT": 420.0})
        log.backfill_prices(prices, "2026-02-03")
        e = log.entries[0]
        assert e.vol_adjusted_return_5d is not None
        # expected_5d_vol = 0.30 * sqrt(5/252) ≈ 0.0423
        expected_vol = 0.30 * np.sqrt(5.0 / 252.0)
        assert abs(e.vol_adjusted_return_5d - (0.05 / expected_vol)) < 1e-4

    def test_selection_alpha(self):
        """Backfill computes selection alpha vs screened average."""
        log = ActionRewardLog()
        e = _buy_entry(date="2026-01-27", price_at_action=400.0,
                       screened_avg_pred_return=0.03)
        log.append_batch([e])
        prices = pd.Series({"MSFT": 420.0})  # +5%
        log.backfill_prices(prices, "2026-02-03")
        e = log.entries[0]
        assert e.selection_alpha_5d is not None
        # selection_alpha = actual_return - screened_avg_pred = 0.05 - 0.03 = 0.02
        assert abs(e.selection_alpha_5d - 0.02) < 1e-6

    def test_confidence_calibration_correct(self):
        """Confidence calibration positive when prediction direction is correct."""
        log = ActionRewardLog()
        e = _buy_entry(date="2026-01-27", price_at_action=400.0,
                       predicted_return=0.08, confidence=0.7)
        log.append_batch([e])
        prices = pd.Series({"MSFT": 420.0})  # Positive return, prediction was positive
        log.backfill_prices(prices, "2026-02-03")
        e = log.entries[0]
        assert e.confidence_calibration is not None
        assert e.confidence_calibration == 0.7  # Correct direction -> +confidence

    def test_confidence_calibration_wrong(self):
        """Confidence calibration negative when prediction direction is wrong."""
        log = ActionRewardLog()
        e = _buy_entry(date="2026-01-27", price_at_action=400.0,
                       predicted_return=0.08, confidence=0.7)
        log.append_batch([e])
        prices = pd.Series({"MSFT": 380.0})  # Negative return, prediction was positive
        log.backfill_prices(prices, "2026-02-03")
        e = log.entries[0]
        assert e.confidence_calibration is not None
        assert e.confidence_calibration == -0.7  # Wrong direction -> -confidence


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

class TestScoreAction:
    def test_returns_tuple(self):
        """score_action now returns (total, components)."""
        e = _sell_entry(entry_price=100.0, price_at_action=110.0)
        e.return_5d = -0.05
        result = score_action(e)
        assert isinstance(result, tuple)
        assert len(result) == 2
        total, components = result
        assert isinstance(total, float)
        assert isinstance(components, dict)

    def test_components_keys(self):
        """All expected component keys are present."""
        e = _sell_entry(entry_price=100.0, price_at_action=110.0)
        e.return_5d = -0.05
        _, components = score_action(e)
        expected_keys = {"base_outcome", "opportunity_cost", "rotation_alpha",
                         "selection_quality", "confidence_calibration",
                         "vol_adjusted", "timing_bonus"}
        assert set(components.keys()) == expected_keys

    # ---- SELL scenarios ----

    def test_sell_profitable_stock_dropped(self):
        """Sold at profit, stock dropped after -> high reward."""
        e = _sell_entry(entry_price=100.0, price_at_action=110.0)
        e.return_5d = -0.05
        total, comps = score_action(e)
        assert comps["base_outcome"] == pytest.approx(0.10, abs=1e-6)
        assert comps["opportunity_cost"] > 0  # Correct sell bonus
        assert comps["timing_bonus"] > 0
        assert total > 0

    def test_sell_premature(self):
        """Sold at small profit, stock rallied after -> penalty."""
        e = _sell_entry(entry_price=100.0, price_at_action=102.0)
        e.return_5d = 0.10  # Stock rose 10% after we sold
        total, comps = score_action(e)
        assert comps["base_outcome"] == pytest.approx(0.02, abs=1e-6)
        assert comps["opportunity_cost"] < 0  # Opp cost penalty
        assert comps["timing_bonus"] < 0  # Bad timing

    def test_sell_with_rotation_alpha_positive(self):
        """Sold A, bought B which outperformed A -> positive rotation alpha."""
        e = _sell_entry(entry_price=100.0, price_at_action=110.0)
        e.return_5d = -0.03
        e.rotation_alpha = 0.08  # Replacement did 8% better
        total, comps = score_action(e)
        assert comps["rotation_alpha"] == pytest.approx(0.08, abs=1e-6)

    def test_sell_with_rotation_alpha_negative(self):
        """Sold A, bought B which underperformed A -> negative rotation alpha."""
        e = _sell_entry(entry_price=100.0, price_at_action=110.0)
        e.return_5d = 0.05
        e.rotation_alpha = -0.10  # Replacement did 10% worse
        total, comps = score_action(e)
        assert comps["rotation_alpha"] == pytest.approx(-0.10, abs=1e-6)

    def test_sell_partial_uses_entry_price_for_base_outcome(self):
        e = _sell_entry(action="SELL_PARTIAL", entry_price=100.0, price_at_action=110.0)
        e.return_5d = -0.01
        _, comps = score_action(e)
        assert comps["base_outcome"] == pytest.approx(0.10, abs=1e-6)

    def test_sell_no_post_return(self):
        """No post-action data yet -> only base gain matters."""
        e = _sell_entry(entry_price=100.0, price_at_action=105.0)
        total, comps = score_action(e)
        assert comps["base_outcome"] == pytest.approx(0.05, abs=1e-6)
        assert comps["opportunity_cost"] == 0.0

    # ---- BUY scenarios ----

    def test_buy_good(self):
        """Bought and stock went up -> positive reward."""
        e = _buy_entry()
        e.return_5d = 0.08
        total, comps = score_action(e)
        assert comps["base_outcome"] == pytest.approx(0.08, abs=1e-6)
        assert comps["timing_bonus"] > 0
        assert total > 0

    def test_buy_bad(self):
        """Bought and stock crashed -> negative reward with extra penalty."""
        e = _buy_entry()
        e.return_5d = -0.10
        total, comps = score_action(e)
        assert comps["base_outcome"] == pytest.approx(-0.10, abs=1e-6)
        assert comps["opportunity_cost"] < 0  # Extra bad-buy penalty
        assert comps["timing_bonus"] < 0
        assert total < 0

    def test_buy_with_selection_alpha(self):
        """Bought the best screened stock -> positive selection alpha."""
        e = _buy_entry()
        e.return_5d = 0.08
        e.selection_alpha_5d = 0.05  # Beat the screened average by 5%
        total, comps = score_action(e)
        assert comps["selection_quality"] == pytest.approx(0.05, abs=1e-6)

    def test_buy_no_post_return(self):
        """No post data -> zero reward."""
        e = _buy_entry()
        total, comps = score_action(e)
        assert total == 0.0

    def test_buy_with_rotation_alpha(self):
        """BUY that replaced a SELL: our buy outperformed the sold stock."""
        e = _buy_entry()
        e.return_5d = 0.05
        e.rotation_alpha = 0.08  # We outperformed the stock we sold by 8%
        total, comps = score_action(e)
        assert comps["rotation_alpha"] == pytest.approx(0.08, abs=1e-6)

    # ---- HOLD scenarios ----

    def test_hold_good(self):
        """Held and stock continued up -> positive reward."""
        e = _hold_entry()
        e.return_5d = 0.05
        total, comps = score_action(e)
        assert comps["base_outcome"] == pytest.approx(0.05, abs=1e-6)
        assert total > 0

    def test_hold_wrong(self):
        """Held through a steep decline -> negative with extra penalty."""
        e = _hold_entry()
        e.return_5d = -0.08
        total, comps = score_action(e)
        assert comps["base_outcome"] == pytest.approx(-0.08, abs=1e-6)
        assert comps["opportunity_cost"] < 0  # Extra penalty for wrong hold

    def test_hold_patient_recovery(self):
        """Held through dip that recovered -> timing bonus."""
        e = _hold_entry()
        e.return_1d = -0.03  # Down on day 1
        e.return_5d = 0.04   # Recovered by day 5
        total, comps = score_action(e)
        assert comps["timing_bonus"] > 0  # Patient hold rewarded
        assert comps["base_outcome"] > 0

    def test_hold_weaker_than_alternative(self):
        """Held a stock when a better screened alternative existed."""
        e = _hold_entry(predicted_return=0.02)
        e.return_5d = 0.01
        e.screened_top_pred_return = 0.08  # Much better option was available
        total, comps = score_action(e)
        # screened_top - predicted = 0.08 - 0.02 = 0.06 > 0.02 threshold
        assert comps["selection_quality"] < 0

    def test_hold_best_available(self):
        """Held the best stock among screened -> no selection penalty."""
        e = _hold_entry(predicted_return=0.08)
        e.return_5d = 0.05
        e.screened_top_pred_return = 0.09  # Only 1% gap, below 2% threshold
        total, comps = score_action(e)
        assert comps["selection_quality"] == 0.0  # No penalty

    # ---- Confidence calibration ----

    def test_confidence_calibration_correct_prediction(self):
        """High confidence + correct direction -> positive calibration component."""
        e = _buy_entry(confidence=0.9, predicted_return=0.05)
        e.return_5d = 0.03  # Positive, matching prediction
        e.confidence_calibration = 0.9
        _, comps = score_action(e)
        assert comps["confidence_calibration"] == pytest.approx(0.9, abs=1e-6)

    def test_confidence_calibration_wrong_prediction(self):
        """High confidence + wrong direction -> negative calibration."""
        e = _buy_entry(confidence=0.9, predicted_return=0.05)
        e.return_5d = -0.03  # Negative, opposite of prediction
        e.confidence_calibration = -0.9
        _, comps = score_action(e)
        assert comps["confidence_calibration"] == pytest.approx(-0.9, abs=1e-6)

    def test_confidence_calibration_fallback(self):
        """When confidence_calibration not pre-computed, falls back to inline calc."""
        e = _buy_entry(confidence=0.8, predicted_return=0.05)
        e.return_5d = 0.02  # Correct direction
        # Don't set e.confidence_calibration -> uses fallback
        _, comps = score_action(e)
        assert comps["confidence_calibration"] == pytest.approx(0.8, abs=1e-6)

    # ---- Vol-adjusted ----

    def test_vol_adjusted_return(self):
        """Vol-adjusted component uses pre-computed value."""
        e = _buy_entry()
        e.return_5d = 0.05
        e.vol_adjusted_return_5d = 1.5  # 1.5 sigma move
        _, comps = score_action(e)
        assert comps["vol_adjusted"] == pytest.approx(1.5 * 0.01, abs=1e-6)

    def test_vol_adjusted_fallback(self):
        """Vol-adjusted falls back to inline computation."""
        e = _buy_entry(stock_volatility=0.30)
        e.return_5d = 0.05
        _, comps = score_action(e)
        expected_5d_vol = 0.30 * np.sqrt(5.0 / 252.0)
        assert comps["vol_adjusted"] == pytest.approx(0.05 / expected_5d_vol * 0.01, abs=1e-4)

    # ---- Custom params ----

    def test_custom_params(self):
        """ActionScoreParams overrides thresholds."""
        e = _buy_entry()
        e.return_5d = -0.02  # Above default bad_buy of -0.03
        params = ActionScoreParams(bad_buy_threshold=-0.01)  # Tighter threshold
        total, comps = score_action(e, params=params)
        # With threshold=-0.01, -0.02 is bad -> extra penalty
        assert comps["opportunity_cost"] < 0

    def test_custom_weights(self):
        """Custom component weights change the total."""
        e = _sell_entry(entry_price=100.0, price_at_action=110.0)
        e.return_5d = -0.05
        # Give all weight to base_outcome
        params = ActionScoreParams(component_weights={
            "base_outcome": 1.0, "opportunity_cost": 0.0,
            "rotation_alpha": 0.0, "selection_quality": 0.0,
            "confidence_calibration": 0.0, "vol_adjusted": 0.0,
            "timing_bonus": 0.0,
        })
        total, comps = score_action(e, params=params)
        assert total == pytest.approx(comps["base_outcome"], abs=1e-6)


class TestScoreActions:
    def test_scores_unscored_only(self):
        log = ActionRewardLog()
        e1 = _sell_entry(ticker="A")
        e1.return_5d = -0.02
        e2 = _buy_entry(ticker="B")
        e2.return_5d = 0.05
        e2.action_reward = 999.0
        log.append_batch([e1, e2])
        n = score_actions(log, window=60)
        assert n == 1
        assert log.entries[0].action_reward is not None
        assert log.entries[0].reward_components is not None
        assert log.entries[1].action_reward == 999.0

    def test_stores_components(self):
        log = ActionRewardLog()
        e = _buy_entry()
        e.return_5d = 0.04
        log.append_batch([e])
        score_actions(log, window=60)
        assert log.entries[0].reward_components is not None
        assert "base_outcome" in log.entries[0].reward_components

    def test_with_custom_params(self):
        log = ActionRewardLog()
        e = _buy_entry()
        e.return_5d = -0.02
        log.append_batch([e])
        params = ActionScoreParams(bad_buy_threshold=-0.01)
        score_actions(log, window=60, params=params)
        assert log.entries[0].action_reward is not None


# ---------------------------------------------------------------------------
# Quality summary
# ---------------------------------------------------------------------------

class TestComputeActionQualitySummary:
    def _build_log(self) -> ActionRewardLog:
        log = ActionRewardLog()
        entries = []
        for i in range(10):
            e = _buy_entry(date=f"2026-02-{i+1:02d}", ticker=f"B{i}")
            ret = 0.05 if i % 2 == 0 else -0.05
            e.return_5d = ret
            e.buy_outcome_return = ret
            e.was_bad_buy = ret < -0.03
            entries.append(e)
        for i in range(5):
            e = _sell_entry(date=f"2026-02-{i+1:02d}", ticker=f"S{i}")
            ret = 0.06 if i % 2 == 0 else -0.02
            e.return_5d = ret
            e.post_sell_return_5d = ret
            e.was_premature_sell = ret > 0.03
            e.entry_price = 100.0
            e.price_at_action = 110.0
            entries.append(e)
        log.append_batch(entries)
        score_actions(log, window=60)
        return log

    def test_summary_structure(self):
        log = self._build_log()
        summary = compute_action_quality_summary(log, window=60)
        assert "BUY" in summary
        assert "SELL" in summary
        assert "overall" in summary
        assert summary["BUY"]["count"] == 10
        assert summary["SELL"]["count"] == 5

    def test_buy_stats(self):
        log = self._build_log()
        summary = compute_action_quality_summary(log, window=60)
        buy = summary["BUY"]
        assert "bad_buy_count" in buy
        assert "bad_buy_pct" in buy
        assert buy["bad_buy_count"] == 5

    def test_sell_stats(self):
        log = self._build_log()
        summary = compute_action_quality_summary(log, window=60)
        sell = summary["SELL"]
        assert "premature_sell_count" in sell
        assert sell["premature_sell_count"] == 3

    def test_empty_log(self):
        log = ActionRewardLog()
        summary = compute_action_quality_summary(log, window=30)
        assert summary == {}

    def test_per_component_averages(self):
        """Summary includes avg_components breakdown."""
        log = self._build_log()
        summary = compute_action_quality_summary(log, window=60)
        buy = summary["BUY"]
        assert "avg_components" in buy
        assert "base_outcome" in buy["avg_components"]

    def test_overall_components(self):
        """Overall summary includes avg_components."""
        log = self._build_log()
        summary = compute_action_quality_summary(log, window=60)
        assert "avg_components" in summary["overall"]

    def test_rotation_stats_in_sell(self):
        """Rotation stats appear when sells have rotation data."""
        log = ActionRewardLog()
        entries = []
        for i in range(3):
            e = _sell_entry(date=f"2026-02-{i+1:02d}", ticker=f"R{i}",
                            entry_price=100.0, price_at_action=110.0)
            e.return_5d = -0.02
            e.rotation_alpha = 0.05 if i < 2 else -0.03
            e.was_premature_sell = False
            entries.append(e)
        log.append_batch(entries)
        score_actions(log, window=60)
        summary = compute_action_quality_summary(log, window=60)
        sell = summary["SELL"]
        assert "rotation_count" in sell
        assert sell["rotation_count"] == 3
        assert "avg_rotation_alpha" in sell
        assert "good_rotation_pct" in sell

    def test_patient_hold_stats(self):
        """Patient hold stats appear when holds had drawdown recovery."""
        log = ActionRewardLog()
        entries = []
        for i in range(4):
            e = _hold_entry(date=f"2026-02-{i+1:02d}", ticker=f"H{i}")
            if i < 2:
                # Patient hold: dipped then recovered
                e.return_1d = -0.02
                e.return_5d = 0.03
            else:
                # Regular hold
                e.return_1d = 0.01
                e.return_5d = 0.02
            e.hold_period_return = e.return_5d
            entries.append(e)
        log.append_batch(entries)
        score_actions(log, window=60)
        summary = compute_action_quality_summary(log, window=60)
        hold = summary["HOLD"]
        assert "patient_hold_count" in hold
        assert hold["patient_hold_count"] == 2
        assert hold["patient_hold_pct"] == 0.5

    def test_confidence_calibration_in_overall(self):
        """Overall summary includes confidence calibration metrics."""
        log = ActionRewardLog()
        entries = []
        for i in range(4):
            e = _buy_entry(date=f"2026-02-{i+1:02d}", ticker=f"C{i}",
                           confidence=0.8, predicted_return=0.05)
            e.return_5d = 0.03 if i < 3 else -0.03  # 3 correct, 1 wrong
            e.confidence_calibration = 0.8 if i < 3 else -0.8
            entries.append(e)
        log.append_batch(entries)
        score_actions(log, window=60)
        summary = compute_action_quality_summary(log, window=60)
        assert "confidence_calibration" in summary["overall"]
        cal = summary["overall"]["confidence_calibration"]
        assert cal["well_calibrated_pct"] == pytest.approx(0.75, abs=1e-6)


class TestBestPostReturn:
    def test_prefers_5d(self):
        e = _buy_entry()
        e.return_1d = 0.01
        e.return_3d = 0.02
        e.return_5d = 0.03
        assert _best_post_return(e) == 0.03

    def test_falls_back_3d(self):
        e = _buy_entry()
        e.return_1d = 0.01
        e.return_3d = 0.02
        assert _best_post_return(e) == 0.02

    def test_falls_back_1d(self):
        e = _buy_entry()
        e.return_1d = 0.01
        assert _best_post_return(e) == 0.01

    def test_none(self):
        e = _buy_entry()
        assert _best_post_return(e) is None

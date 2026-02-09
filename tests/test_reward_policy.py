"""Tests for the reward policy (Bayesian linear bandit) and feedback engine."""
import json
import numpy as np
import pandas as pd
import pytest

from stock_screener.reward.policy import (
    RewardPolicy,
    PolicyState,
    build_state_vector,
    compute_equity_slope,
    compute_recent_sharpe,
    compute_reward,
)
from stock_screener.reward.feedback import (
    compute_online_ic,
    compute_ensemble_reward_weights,
    build_verified_labels,
    compute_prediction_bias,
)
from stock_screener.reward.tracker import RewardEntry, RewardLog


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------

class TestBuildStateVector:
    def test_shape(self):
        s = build_state_vector()
        assert s.shape == (8,)

    def test_clipping(self):
        s = build_state_vector(portfolio_drawdown=-5.0, equity_slope_5d=99.0)
        assert s[0] == -1.0  # Clipped to -1
        assert s[1] == 1.0   # Clipped to 1


class TestComputeReward:
    def test_positive_return(self):
        r = compute_reward(0.02, drawdown_penalty_lambda=2.0)
        assert r == 0.02  # No penalty for gains

    def test_negative_return(self):
        r = compute_reward(-0.05, drawdown_penalty_lambda=2.0)
        expected = -0.05 - 2.0 * 0.05**2
        assert abs(r - expected) < 1e-9

    def test_zero_return(self):
        assert compute_reward(0.0) == 0.0


class TestRewardPolicy:
    def test_warmup_returns_default(self):
        policy = RewardPolicy(warmup_days=10)
        state = build_state_vector()
        action = policy.select_action(state)
        assert action["exposure_scalar"] == 1.0
        assert action["conviction_scalar"] == 1.0

    def test_update_increments_count(self):
        policy = RewardPolicy(warmup_days=5)
        state = build_state_vector()
        for i in range(5):
            policy.update(state, {"exposure_scalar": 1.0, "conviction_scalar": 1.0}, 0.01)
        assert policy.state.n_updates == 5
        assert policy.is_warm

    def test_after_warmup_selects_action(self):
        policy = RewardPolicy(warmup_days=3)
        state = build_state_vector()
        # Warm up
        for _ in range(5):
            policy.update(state, {"exposure_scalar": 1.0, "conviction_scalar": 1.0}, 0.01)
        action = policy.select_action(state)
        assert policy.exposure_min <= action["exposure_scalar"] <= policy.exposure_max
        assert policy.conviction_min <= action["conviction_scalar"] <= policy.conviction_max

    def test_save_load_roundtrip(self, tmp_path):
        policy = RewardPolicy(warmup_days=3)
        state = build_state_vector()
        for _ in range(5):
            policy.update(state, {"exposure_scalar": 1.1, "conviction_scalar": 1.2}, 0.02)

        path = tmp_path / "policy.json"
        policy.save(path)
        loaded = RewardPolicy.load(path, warmup_days=3)
        assert loaded.state.n_updates == 5
        assert loaded.is_warm
        assert abs(loaded.state.cumulative_reward - policy.state.cumulative_reward) < 1e-6

    def test_load_missing_file(self, tmp_path):
        policy = RewardPolicy.load(tmp_path / "nonexistent.json")
        assert policy.state.n_updates == 0

    def test_summary(self):
        policy = RewardPolicy()
        s = policy.summary()
        assert "n_updates" in s
        assert "is_warm" in s

    def test_cumulative_reward_tracks(self):
        policy = RewardPolicy(warmup_days=0)
        state = build_state_vector()
        policy.update(state, {"exposure_scalar": 1.0, "conviction_scalar": 1.0}, 0.05)
        assert policy.state.cumulative_reward > 0


class TestEquitySlope:
    def test_empty_history(self):
        assert compute_equity_slope([]) == 0.0

    def test_single_entry(self):
        assert compute_equity_slope([{"equity_cad": 500}]) == 0.0

    def test_rising_equity(self):
        history = [{"equity_cad": 500 + i * 10} for i in range(5)]
        slope = compute_equity_slope(history, window=5)
        assert slope > 0

    def test_falling_equity(self):
        history = [{"equity_cad": 500 - i * 10} for i in range(5)]
        slope = compute_equity_slope(history, window=5)
        assert slope < 0


class TestRecentSharpe:
    def test_empty(self):
        assert compute_recent_sharpe([]) == 0.0

    def test_constant_equity(self):
        history = [{"equity_cad": 500} for _ in range(5)]
        assert compute_recent_sharpe(history) == 0.0

    def test_positive_trend(self):
        history = [{"equity_cad": 500 + i * 5} for i in range(10)]
        sharpe = compute_recent_sharpe(history, window=5)
        assert sharpe > 0


# ---------------------------------------------------------------------------
# Feedback tests
# ---------------------------------------------------------------------------

def _make_log_with_data(n: int = 20) -> RewardLog:
    """Create a reward log with n entries that have predictions and realized returns."""
    log = RewardLog()
    rng = np.random.RandomState(42)
    for i in range(n):
        pred = float(rng.randn() * 0.05)
        realized = pred * 0.5 + float(rng.randn() * 0.02)  # Correlated
        log.append(RewardEntry(
            date=f"2026-02-{i+1:02d}" if i < 28 else f"2026-01-{i-27:02d}",
            ticker=f"T{i}",
            predicted_return=pred,
            realized_1d_return=realized,
            price_at_prediction=100.0,
            price_next_day=100.0 * (1 + realized),
            per_model_preds=[pred + float(rng.randn() * 0.01) for _ in range(3)],
        ))
    return log


class TestComputeOnlineIC:
    def test_too_few_entries(self):
        log = RewardLog()
        log.append(RewardEntry(date="2026-02-01", ticker="A", predicted_return=0.01, realized_1d_return=0.01))
        result = compute_online_ic(log, window=30)
        assert result["ensemble_ic"] is None
        assert result["n_observations"] == 1

    def test_with_data(self):
        log = _make_log_with_data(20)
        result = compute_online_ic(log, window=60)
        assert result["ensemble_ic"] is not None
        assert result["n_observations"] >= 5
        # With correlated pred/realized, IC should be positive
        assert result["ensemble_ic"] > 0

    def test_per_model_ics(self):
        log = _make_log_with_data(20)
        result = compute_online_ic(log, window=60)
        assert result["per_model_ics"] is not None
        assert len(result["per_model_ics"]) == 3


class TestComputeEnsembleRewardWeights:
    def test_equal_ics(self):
        weights = compute_ensemble_reward_weights(
            [0.1, 0.1, 0.1],
            holdout_weights=[0.33, 0.33, 0.34],
        )
        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_one_dominant(self):
        weights = compute_ensemble_reward_weights(
            [0.5, 0.0, 0.0],
            holdout_weights=[0.33, 0.33, 0.34],
            blend_alpha=1.0,  # Fully realized
        )
        assert weights[0] > 0.9  # Dominant model gets most weight

    def test_blend_alpha_zero(self):
        """blend_alpha=0 should return holdout weights."""
        holdout = [0.5, 0.3, 0.2]
        weights = compute_ensemble_reward_weights(
            [0.1, 0.2, 0.3],
            holdout_weights=holdout,
            blend_alpha=0.0,
        )
        for w, h in zip(weights, holdout):
            assert abs(w - h) < 1e-6

    def test_all_negative_ics(self):
        """If all ICs are negative, should fall back to equal weights for realized portion."""
        weights = compute_ensemble_reward_weights(
            [-0.1, -0.2, -0.3],
            holdout_weights=[0.5, 0.3, 0.2],
            blend_alpha=0.5,
        )
        assert abs(sum(weights) - 1.0) < 1e-6


class TestBuildVerifiedLabels:
    def test_empty_log(self):
        log = RewardLog()
        df = build_verified_labels(log)
        assert df.empty

    def test_with_closed_trades(self):
        log = RewardLog()
        log.append(RewardEntry(
            date="2026-02-01", ticker="AAPL", predicted_return=0.05,
            realized_cumulative_return=0.08, days_held=5, exit_reason="TAKE_PROFIT",
        ))
        log.append(RewardEntry(
            date="2026-02-01", ticker="MSFT", predicted_return=0.03,
        ))  # No exit -> not included
        df = build_verified_labels(log)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[0]["realized_return"] == 0.08


class TestComputePredictionBias:
    def test_too_few(self):
        log = RewardLog()
        result = compute_prediction_bias(log)
        assert result["mean_bias"] == 0.0

    def test_with_data(self):
        log = _make_log_with_data(20)
        result = compute_prediction_bias(log, window=60)
        assert "mean_bias" in result
        assert result["n_observations"] >= 5

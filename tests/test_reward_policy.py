"""Tests for the reward policy (Bayesian linear bandit) and feedback engine."""
import json
import numpy as np
import pandas as pd
import pytest

from stock_screener.reward.policy import (
    RewardPolicy,
    PolicyState,
    ACTION_NAMES,
    N_ACTIONS,
    _DEFAULT_ACTION,
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


# Convenience: full 4D action dict with defaults
def _action(**overrides: float) -> dict[str, float]:
    a = dict(_DEFAULT_ACTION)
    a.update(overrides)
    return a


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------

class TestBuildStateVector:
    def test_shape(self):
        s = build_state_vector()
        assert s.shape == (8,)

    def test_clipping(self):
        s = build_state_vector(portfolio_drawdown=-5.0, equity_slope_5d=99.0)
        assert s[0] == -1.0
        assert s[1] == 1.0


class TestComputeReward:
    def test_positive_return(self):
        r = compute_reward(0.02, drawdown_penalty_lambda=2.0)
        assert r == 0.02

    def test_negative_return(self):
        r = compute_reward(-0.05, drawdown_penalty_lambda=2.0)
        expected = -0.05 - 2.0 * 0.05**2
        assert abs(r - expected) < 1e-9

    def test_zero_return(self):
        assert compute_reward(0.0) == 0.0


class TestActionConstants:
    def test_n_actions(self):
        assert N_ACTIONS == 4

    def test_action_names(self):
        assert ACTION_NAMES == [
            "exposure_scalar", "conviction_scalar",
            "exit_tightness", "hold_patience",
        ]

    def test_default_action(self):
        for name in ACTION_NAMES:
            assert _DEFAULT_ACTION[name] == 1.0


class TestPolicyState:
    def test_dim_includes_all_actions(self):
        ps = PolicyState()
        # 8 state features + 4 actions + 1 bias
        assert ps.dim == 13

    def test_mu_length(self):
        ps = PolicyState()
        assert len(ps.mu) == 13

    def test_precision_shape(self):
        ps = PolicyState()
        P = ps.precision_matrix
        assert P.shape == (13, 13)


class TestRewardPolicy:
    def test_warmup_returns_default(self):
        policy = RewardPolicy(warmup_days=10)
        state = build_state_vector()
        action = policy.select_action(state)
        for name in ACTION_NAMES:
            assert action[name] == 1.0

    def test_update_increments_count(self):
        policy = RewardPolicy(warmup_days=5)
        state = build_state_vector()
        for i in range(5):
            policy.update(state, _action(), 0.01)
        assert policy.state.n_updates == 5
        assert policy.is_warm

    def test_after_warmup_selects_action(self):
        policy = RewardPolicy(warmup_days=3)
        state = build_state_vector()
        for _ in range(5):
            policy.update(state, _action(), 0.01)
        action = policy.select_action(state)
        assert policy.exposure_min <= action["exposure_scalar"] <= policy.exposure_max
        assert policy.conviction_min <= action["conviction_scalar"] <= policy.conviction_max
        assert policy.exit_tightness_min <= action["exit_tightness"] <= policy.exit_tightness_max
        assert policy.hold_patience_min <= action["hold_patience"] <= policy.hold_patience_max

    def test_all_action_keys_present(self):
        policy = RewardPolicy(warmup_days=2)
        state = build_state_vector()
        for _ in range(3):
            policy.update(state, _action(), 0.01)
        action = policy.select_action(state)
        for name in ACTION_NAMES:
            assert name in action

    def test_save_load_roundtrip(self, tmp_path):
        policy = RewardPolicy(warmup_days=3, exit_tightness_min=0.6, hold_patience_max=1.8)
        state = build_state_vector()
        for _ in range(5):
            policy.update(state, _action(exposure_scalar=1.1, conviction_scalar=1.2), 0.02)

        path = tmp_path / "policy.json"
        policy.save(path)
        loaded = RewardPolicy.load(path, warmup_days=3)
        assert loaded.state.n_updates == 5
        assert loaded.is_warm
        assert abs(loaded.state.cumulative_reward - policy.state.cumulative_reward) < 1e-6
        assert loaded.exit_tightness_min == 0.6
        assert loaded.hold_patience_max == 1.8

    def test_load_missing_file(self, tmp_path):
        policy = RewardPolicy.load(tmp_path / "nonexistent.json")
        assert policy.state.n_updates == 0

    def test_dimension_migration(self, tmp_path):
        """Loading an old 2-action policy file should migrate gracefully."""
        # Simulate an old policy file with dim=11 (8 state + 2 action + 1 bias)
        old_dim = 11
        old_mu = [0.1] * old_dim
        old_precision = (np.eye(old_dim) * 0.25).flatten().tolist()
        old_data = {
            "config": {"warmup_days": 5},
            "policy_state": {
                "dim": old_dim,
                "mu": old_mu,
                "precision_flat": old_precision,
                "a": 2.0,
                "b": 1.5,
                "n_updates": 30,
                "cumulative_reward": 0.5,
                "history": [{"exposure": 1.1, "conviction": 1.2, "reward": 0.01, "daily_return": 0.01}],
            },
        }
        path = tmp_path / "old_policy.json"
        path.write_text(json.dumps(old_data))
        loaded = RewardPolicy.load(path, warmup_days=5)
        # Should preserve bookkeeping but reset posterior for new dim
        assert loaded.state.n_updates == 30
        assert loaded.state.dim == 13  # New dim: 8 + 4 + 1
        assert len(loaded.state.mu) == 13
        assert loaded.state.precision_matrix.shape == (13, 13)

    def test_summary(self):
        policy = RewardPolicy()
        s = policy.summary()
        assert "n_updates" in s
        assert "is_warm" in s

    def test_summary_with_history(self):
        policy = RewardPolicy(warmup_days=1)
        state = build_state_vector()
        policy.update(state, _action(exit_tightness=1.5), 0.02)
        s = policy.summary()
        assert "last_action" in s
        assert s["last_action"]["exit_tightness"] == 1.5

    def test_cumulative_reward_tracks(self):
        policy = RewardPolicy(warmup_days=0)
        state = build_state_vector()
        policy.update(state, _action(), 0.05)
        assert policy.state.cumulative_reward > 0

    def test_history_stores_all_actions(self):
        policy = RewardPolicy(warmup_days=0)
        state = build_state_vector()
        policy.update(
            state,
            _action(exposure_scalar=1.2, exit_tightness=0.8, hold_patience=1.5),
            0.01,
        )
        h = policy.state.history[-1]
        assert h["exposure_scalar"] == 1.2
        assert h["exit_tightness"] == 0.8
        assert h["hold_patience"] == 1.5

    def test_update_with_partial_action_dict(self):
        """Old-style 2-key action dict should work (missing keys default to 1.0)."""
        policy = RewardPolicy(warmup_days=0)
        state = build_state_vector()
        # Only exposure and conviction — should not crash
        policy.update(state, {"exposure_scalar": 1.1, "conviction_scalar": 1.0}, 0.01)
        assert policy.state.n_updates == 1
        h = policy.state.history[-1]
        assert h["exit_tightness"] == 1.0
        assert h["hold_patience"] == 1.0


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
        realized = pred * 0.5 + float(rng.randn() * 0.02)
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
            blend_alpha=1.0,
        )
        assert weights[0] > 0.9

    def test_blend_alpha_zero(self):
        holdout = [0.5, 0.3, 0.2]
        weights = compute_ensemble_reward_weights(
            [0.1, 0.2, 0.3],
            holdout_weights=holdout,
            blend_alpha=0.0,
        )
        for w, h in zip(weights, holdout):
            assert abs(w - h) < 1e-6

    def test_all_negative_ics(self):
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
        ))
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

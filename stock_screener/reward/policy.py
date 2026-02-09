"""Adaptive exposure policy using Bayesian linear Thompson Sampling.

The policy learns the relationship between portfolio/market state and
optimal portfolio scalars, using daily returns as reward.  No
deep-learning dependencies -- uses only numpy for closed-form
Bayesian linear regression updates.

Action dimensions (4):
  0: exposure_scalar     — total portfolio exposure level
  1: conviction_scalar   — amplify spread between high/low conviction picks
  2: exit_tightness      — scale stop-loss distances (high = tighter stops)
  3: hold_patience        — scale holding period (high = hold longer)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# State feature names (order matters -- must match build_state_vector)
STATE_FEATURES = [
    "portfolio_drawdown",
    "equity_slope_5d",
    "regime_composite",
    "model_avg_confidence",
    "prediction_spread",
    "n_positions_norm",
    "recent_sharpe_5d",
    "reward_ic_recent",
]

# Action dimension names (order matters)
ACTION_NAMES = [
    "exposure_scalar",
    "conviction_scalar",
    "exit_tightness",
    "hold_patience",
]
N_ACTIONS = len(ACTION_NAMES)

# Default action values (used during warmup and fallback)
_DEFAULT_ACTION: dict[str, float] = {
    "exposure_scalar": 1.0,
    "conviction_scalar": 1.0,
    "exit_tightness": 1.0,
    "hold_patience": 1.0,
}


@dataclass
class PolicyState:
    """Bayesian linear regression posterior parameters.

    Model: reward = phi(state, action)^T @ w + noise
    Prior: w ~ N(mu, (1/a) * Sigma), noise ~ N(0, 1/b)
    Posterior updated via closed-form Bayesian linear regression.
    """

    # Dimension = state_dim + N_ACTIONS + 1 (bias)
    dim: int = len(STATE_FEATURES) + N_ACTIONS + 1

    # Posterior mean of weights
    mu: list[float] = field(default_factory=list)
    # Posterior precision matrix (inverse covariance), stored as flat list
    precision_flat: list[float] = field(default_factory=list)
    # Noise precision parameters (Gamma distribution)
    a: float = 1.0  # shape
    b: float = 1.0  # rate

    # Bookkeeping
    n_updates: int = 0
    cumulative_reward: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.mu:
            self.mu = [0.0] * self.dim
        if not self.precision_flat:
            # Identity * lambda_prior (regularization)
            precision = np.eye(self.dim, dtype=float) * 0.25
            self.precision_flat = precision.flatten().tolist()

    @property
    def mu_array(self) -> np.ndarray:
        return np.array(self.mu, dtype=float)

    @property
    def precision_matrix(self) -> np.ndarray:
        return np.array(self.precision_flat, dtype=float).reshape(self.dim, self.dim)


def build_state_vector(
    portfolio_drawdown: float = 0.0,
    equity_slope_5d: float = 0.0,
    regime_composite: float = 0.0,
    model_avg_confidence: float = 0.5,
    prediction_spread: float = 0.0,
    n_positions: int = 0,
    recent_sharpe_5d: float = 0.0,
    reward_ic_recent: float = 0.0,
) -> np.ndarray:
    """Construct the state vector from portfolio/market features."""
    return np.array([
        np.clip(portfolio_drawdown, -1.0, 0.0),
        np.clip(equity_slope_5d, -1.0, 1.0),
        np.clip(regime_composite, -1.0, 1.0),
        np.clip(model_avg_confidence, 0.0, 1.0),
        np.clip(prediction_spread, 0.0, 2.0),
        np.clip(n_positions / 10.0, 0.0, 1.0),  # Normalized
        np.clip(recent_sharpe_5d, -3.0, 3.0),
        np.clip(reward_ic_recent, -1.0, 1.0),
    ], dtype=float)


def _build_feature_vector(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Combine state and action into a feature vector with bias term."""
    return np.concatenate([state, action, [1.0]])


def compute_reward(
    daily_return: float,
    drawdown_penalty_lambda: float = 2.0,
) -> float:
    """Asymmetric reward: penalises large losses more than it rewards gains."""
    penalty = drawdown_penalty_lambda * max(0.0, -daily_return) ** 2
    return daily_return - penalty


class RewardPolicy:
    """Bayesian linear bandit for adaptive portfolio parameter scaling.

    Controls 4 scalars: exposure, conviction spread, exit tightness,
    and hold patience.
    """

    def __init__(
        self,
        warmup_days: int = 20,
        exposure_min: float = 0.3,
        exposure_max: float = 1.5,
        conviction_min: float = 0.5,
        conviction_max: float = 2.0,
        exit_tightness_min: float = 0.5,
        exit_tightness_max: float = 2.0,
        hold_patience_min: float = 0.5,
        hold_patience_max: float = 2.0,
        drawdown_penalty: float = 2.0,
        exploration_scale: float = 0.5,
    ) -> None:
        self.warmup_days = warmup_days
        self.exposure_min = exposure_min
        self.exposure_max = exposure_max
        self.conviction_min = conviction_min
        self.conviction_max = conviction_max
        self.exit_tightness_min = exit_tightness_min
        self.exit_tightness_max = exit_tightness_max
        self.hold_patience_min = hold_patience_min
        self.hold_patience_max = hold_patience_max
        self.drawdown_penalty = drawdown_penalty
        self.exploration_scale = exploration_scale
        self.state = PolicyState()

    # ---- persistence --------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "RewardPolicy":
        """Load policy from JSON or create a fresh one.

        Handles migration from older 2-action policies gracefully by
        resetting the posterior when the dimension has changed.
        """
        p = Path(path)
        policy = cls(**kwargs)
        if not p.exists():
            return policy
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return policy

        ps = data.get("policy_state", {})

        # Dimension migration: if saved dim differs from current, keep
        # bookkeeping (n_updates, history) but reset the posterior.
        saved_dim = ps.get("dim", policy.state.dim)
        if saved_dim != policy.state.dim:
            policy.state.n_updates = ps.get("n_updates", 0)
            policy.state.cumulative_reward = ps.get("cumulative_reward", 0.0)
            policy.state.history = ps.get("history", [])
            # mu and precision stay at fresh defaults for the new dim
        else:
            policy.state = PolicyState(
                dim=saved_dim,
                mu=ps.get("mu", []),
                precision_flat=ps.get("precision_flat", []),
                a=ps.get("a", 1.0),
                b=ps.get("b", 1.0),
                n_updates=ps.get("n_updates", 0),
                cumulative_reward=ps.get("cumulative_reward", 0.0),
                history=ps.get("history", []),
            )

        cfg = data.get("config", {})
        policy.warmup_days = cfg.get("warmup_days", policy.warmup_days)
        policy.exposure_min = cfg.get("exposure_min", policy.exposure_min)
        policy.exposure_max = cfg.get("exposure_max", policy.exposure_max)
        policy.conviction_min = cfg.get("conviction_min", policy.conviction_min)
        policy.conviction_max = cfg.get("conviction_max", policy.conviction_max)
        policy.exit_tightness_min = cfg.get("exit_tightness_min", policy.exit_tightness_min)
        policy.exit_tightness_max = cfg.get("exit_tightness_max", policy.exit_tightness_max)
        policy.hold_patience_min = cfg.get("hold_patience_min", policy.hold_patience_min)
        policy.hold_patience_max = cfg.get("hold_patience_max", policy.hold_patience_max)
        policy.drawdown_penalty = cfg.get("drawdown_penalty", policy.drawdown_penalty)
        return policy

    def save(self, path: str | Path) -> None:
        """Persist policy state to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "warmup_days": self.warmup_days,
                "exposure_min": self.exposure_min,
                "exposure_max": self.exposure_max,
                "conviction_min": self.conviction_min,
                "conviction_max": self.conviction_max,
                "exit_tightness_min": self.exit_tightness_min,
                "exit_tightness_max": self.exit_tightness_max,
                "hold_patience_min": self.hold_patience_min,
                "hold_patience_max": self.hold_patience_max,
                "drawdown_penalty": self.drawdown_penalty,
            },
            "policy_state": {
                "dim": self.state.dim,
                "mu": self.state.mu,
                "precision_flat": self.state.precision_flat,
                "a": self.state.a,
                "b": self.state.b,
                "n_updates": self.state.n_updates,
                "cumulative_reward": self.state.cumulative_reward,
                "history": self.state.history[-60:],
            },
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- action selection ---------------------------------------------------

    def _action_bounds(self) -> list[tuple[float, float]]:
        """Return (min, max) for each action dimension."""
        return [
            (self.exposure_min, self.exposure_max),
            (self.conviction_min, self.conviction_max),
            (self.exit_tightness_min, self.exit_tightness_max),
            (self.hold_patience_min, self.hold_patience_max),
        ]

    def select_action(self, state_vec: np.ndarray) -> dict[str, float]:
        """Select all 4 scalars given current state.

        During warmup returns defaults (all 1.0).  After warmup uses
        Thompson Sampling with grid search over candidate actions.
        """
        if self.state.n_updates < self.warmup_days:
            return dict(_DEFAULT_ACTION)

        # Thompson Sampling: sample weight vector from posterior
        try:
            precision = self.state.precision_matrix
            cov = np.linalg.inv(precision) * (self.state.b / max(self.state.a, 1e-6))
            cov = (cov + cov.T) / 2
            eigvals = np.linalg.eigvalsh(cov)
            if eigvals.min() < 0:
                cov += np.eye(self.state.dim) * (abs(eigvals.min()) + 1e-6)
            w_sample = np.random.multivariate_normal(
                self.state.mu_array, cov * self.exploration_scale,
            )
        except (np.linalg.LinAlgError, ValueError):
            return dict(_DEFAULT_ACTION)

        bounds = self._action_bounds()
        # Grid: 5 points for exposure, 4 for conviction, 3 for exit/hold
        # Total: 5*4*3*3 = 180 candidates — fast enough
        grids = [
            np.linspace(lo, hi, n)
            for (lo, hi), n in zip(bounds, [5, 4, 3, 3])
        ]

        best_reward = -np.inf
        best_action = np.ones(N_ACTIONS)
        for exp_s in grids[0]:
            for conv_s in grids[1]:
                for exit_t in grids[2]:
                    for hold_p in grids[3]:
                        action = np.array([exp_s, conv_s, exit_t, hold_p])
                        phi = _build_feature_vector(state_vec, action)
                        predicted_reward = float(w_sample @ phi)
                        if predicted_reward > best_reward:
                            best_reward = predicted_reward
                            best_action = action

        return {
            name: float(np.clip(best_action[i], bounds[i][0], bounds[i][1]))
            for i, name in enumerate(ACTION_NAMES)
        }

    # ---- posterior update ----------------------------------------------------

    def update(
        self,
        state_vec: np.ndarray,
        action: dict[str, float],
        daily_return: float,
    ) -> None:
        """Update posterior with observed (state, action, reward) tuple."""
        reward = compute_reward(daily_return, self.drawdown_penalty)
        action_vec = np.array([action.get(n, 1.0) for n in ACTION_NAMES])
        phi = _build_feature_vector(state_vec, action_vec)

        # Bayesian linear regression update (rank-1)
        precision = self.state.precision_matrix
        mu = self.state.mu_array

        precision_new = precision + np.outer(phi, phi)
        mu_new = np.linalg.solve(precision_new, precision @ mu + phi * reward)

        self.state.precision_flat = precision_new.flatten().tolist()
        self.state.mu = mu_new.tolist()

        # Update noise precision (simplified: online mean)
        residual = reward - float(mu @ phi)
        self.state.a += 0.5
        self.state.b += 0.5 * residual ** 2

        self.state.n_updates += 1
        self.state.cumulative_reward += reward

        # Compact history entry
        self.state.history.append({
            "reward": float(reward),
            "daily_return": float(daily_return),
            **{n: float(action.get(n, 1.0)) for n in ACTION_NAMES},
        })

    # ---- helpers ------------------------------------------------------------

    @property
    def is_warm(self) -> bool:
        return self.state.n_updates >= self.warmup_days

    def summary(self) -> dict[str, Any]:
        """Return a compact summary for logging / run metadata."""
        result: dict[str, Any] = {
            "n_updates": self.state.n_updates,
            "is_warm": self.is_warm,
            "cumulative_reward": round(self.state.cumulative_reward, 6),
            "avg_reward": round(
                self.state.cumulative_reward / max(self.state.n_updates, 1), 6
            ),
        }
        if self.state.history:
            last = self.state.history[-1]
            result["last_action"] = {
                n: last.get(n, 1.0) for n in ACTION_NAMES
            }
        return result


def compute_equity_slope(pnl_history: list[dict[str, Any]], window: int = 5) -> float:
    """Compute normalised slope of equity over the last *window* snapshots."""
    equities = [
        float(h["equity_cad"])
        for h in pnl_history
        if "equity_cad" in h and h["equity_cad"] is not None and float(h["equity_cad"]) > 0
    ]
    if len(equities) < 2:
        return 0.0
    recent = equities[-window:]
    if len(recent) < 2:
        return 0.0
    x = np.arange(len(recent), dtype=float)
    y = np.array(recent, dtype=float)
    # Normalise by mean equity to get a dimensionless slope
    mean_eq = float(np.mean(y))
    if mean_eq <= 0:
        return 0.0
    # Simple linear regression slope
    slope = float(np.polyfit(x, y / mean_eq, 1)[0])
    return np.clip(slope, -1.0, 1.0)


def compute_recent_sharpe(pnl_history: list[dict[str, Any]], window: int = 5) -> float:
    """Compute Sharpe ratio of daily equity returns over last *window* snapshots."""
    equities = [
        float(h["equity_cad"])
        for h in pnl_history
        if "equity_cad" in h and h["equity_cad"] is not None and float(h["equity_cad"]) > 0
    ]
    if len(equities) < 3:
        return 0.0
    recent = equities[-window:]
    if len(recent) < 2:
        return 0.0
    rets = np.diff(recent) / np.array(recent[:-1])
    std = float(np.std(rets))
    if std < 1e-9:
        return 0.0
    return float(np.mean(rets) / std)

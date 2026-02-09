"""Adaptive exposure policy using Bayesian linear Thompson Sampling.

The policy learns the relationship between portfolio/market state and
optimal exposure/conviction scaling, using daily returns as reward.
No deep-learning dependencies -- uses only numpy for closed-form
Bayesian linear regression updates.
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

# Number of action dimensions
# 0: exposure_scalar, 1: conviction_scalar
N_ACTIONS = 2


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
    """Bayesian linear bandit for adaptive exposure/conviction scaling."""

    def __init__(
        self,
        warmup_days: int = 20,
        exposure_min: float = 0.3,
        exposure_max: float = 1.5,
        conviction_min: float = 0.5,
        conviction_max: float = 2.0,
        drawdown_penalty: float = 2.0,
        exploration_scale: float = 0.5,
    ) -> None:
        self.warmup_days = warmup_days
        self.exposure_min = exposure_min
        self.exposure_max = exposure_max
        self.conviction_min = conviction_min
        self.conviction_max = conviction_max
        self.drawdown_penalty = drawdown_penalty
        self.exploration_scale = exploration_scale
        self.state = PolicyState()

    # ---- persistence --------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "RewardPolicy":
        """Load policy from JSON or create a fresh one."""
        p = Path(path)
        policy = cls(**kwargs)
        if not p.exists():
            return policy
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return policy

        ps = data.get("policy_state", {})
        policy.state = PolicyState(
            dim=ps.get("dim", policy.state.dim),
            mu=ps.get("mu", []),
            precision_flat=ps.get("precision_flat", []),
            a=ps.get("a", 1.0),
            b=ps.get("b", 1.0),
            n_updates=ps.get("n_updates", 0),
            cumulative_reward=ps.get("cumulative_reward", 0.0),
            history=ps.get("history", []),
        )
        # Load config overrides
        cfg = data.get("config", {})
        policy.warmup_days = cfg.get("warmup_days", policy.warmup_days)
        policy.exposure_min = cfg.get("exposure_min", policy.exposure_min)
        policy.exposure_max = cfg.get("exposure_max", policy.exposure_max)
        policy.conviction_min = cfg.get("conviction_min", policy.conviction_min)
        policy.conviction_max = cfg.get("conviction_max", policy.conviction_max)
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
                # Keep only last 60 history entries to limit file size
                "history": self.state.history[-60:],
            },
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- action selection ---------------------------------------------------

    def select_action(self, state_vec: np.ndarray) -> dict[str, float]:
        """Select exposure_scalar and conviction_scalar given current state.

        During warmup returns defaults (1.0, 1.0).  After warmup uses
        Thompson Sampling: sample weights from posterior, pick action that
        maximises predicted reward.
        """
        if self.state.n_updates < self.warmup_days:
            return {"exposure_scalar": 1.0, "conviction_scalar": 1.0}

        # Thompson Sampling: sample weight vector from posterior
        try:
            precision = self.state.precision_matrix
            cov = np.linalg.inv(precision) * (self.state.b / max(self.state.a, 1e-6))
            # Ensure positive semi-definite
            cov = (cov + cov.T) / 2
            eigvals = np.linalg.eigvalsh(cov)
            if eigvals.min() < 0:
                cov += np.eye(self.state.dim) * (abs(eigvals.min()) + 1e-6)
            w_sample = np.random.multivariate_normal(self.state.mu_array, cov * self.exploration_scale)
        except (np.linalg.LinAlgError, ValueError):
            return {"exposure_scalar": 1.0, "conviction_scalar": 1.0}

        # Grid search over candidate actions to find the one with best predicted reward
        best_reward = -np.inf
        best_action = np.array([1.0, 1.0])
        for exp_s in np.linspace(self.exposure_min, self.exposure_max, 7):
            for conv_s in np.linspace(self.conviction_min, self.conviction_max, 5):
                action = np.array([exp_s, conv_s])
                phi = _build_feature_vector(state_vec, action)
                predicted_reward = float(w_sample @ phi)
                if predicted_reward > best_reward:
                    best_reward = predicted_reward
                    best_action = action

        return {
            "exposure_scalar": float(np.clip(best_action[0], self.exposure_min, self.exposure_max)),
            "conviction_scalar": float(np.clip(best_action[1], self.conviction_min, self.conviction_max)),
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
        action_vec = np.array([action["exposure_scalar"], action["conviction_scalar"]])
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

        # Append to history (compact)
        self.state.history.append({
            "reward": float(reward),
            "daily_return": float(daily_return),
            "exposure": action["exposure_scalar"],
            "conviction": action["conviction_scalar"],
        })

    # ---- helpers ------------------------------------------------------------

    @property
    def is_warm(self) -> bool:
        return self.state.n_updates >= self.warmup_days

    def summary(self) -> dict[str, Any]:
        """Return a compact summary for logging / run metadata."""
        return {
            "n_updates": self.state.n_updates,
            "is_warm": self.is_warm,
            "cumulative_reward": round(self.state.cumulative_reward, 6),
            "avg_reward": round(
                self.state.cumulative_reward / max(self.state.n_updates, 1), 6
            ),
        }


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

"""Probability models for EVR."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...types import TradeResult


class RollingBayes:
    """Rolling Bayesian probability estimator with Beta-Binomial smoothing."""
    
    def __init__(
        self,
        window_size: int = 252,
        min_samples: int = 50,
        alpha: float = 0.1,
        beta_prior_alpha: float = 1.0,
        beta_prior_beta: float = 1.0,
        use_regimes: bool = False,
        regime_window: int = 63,
        regime_threshold: float = 0.1,
    ):
        """Initialize RollingBayes.
        
        Args:
            window_size: Rolling window size in trading days
            min_samples: Minimum samples required for estimation
            alpha: EWMA smoothing factor
            beta_prior_alpha: Beta distribution alpha parameter
            beta_prior_beta: Beta distribution beta parameter
            use_regimes: Whether to use regime detection
            regime_window: Window size for regime detection
            regime_threshold: Threshold for regime change detection
        """
        self.window_size = window_size
        self.min_samples = min_samples
        self.alpha = alpha
        self.beta_prior_alpha = beta_prior_alpha
        self.beta_prior_beta = beta_prior_beta
        self.use_regimes = use_regimes
        self.regime_window = regime_window
        self.regime_threshold = regime_threshold
        
        # Internal state
        self._trades: List[TradeResult] = []
        self._regimes: List[str] = []
        self._current_regime: str = "normal"
        self._regime_probabilities: Dict[str, float] = {}
        self._ewma_win_rate: Optional[float] = None
        self._ewma_avg_win: Optional[float] = None
        self._ewma_avg_loss: Optional[float] = None
    
    def update(self, trade: TradeResult) -> None:
        """Update model with new trade result.
        
        Args:
            trade: Trade result to add
        """
        self._trades.append(trade)
        
        # Update regime if enabled
        if self.use_regimes:
            self._update_regime(trade)
        
        # Update EWMA estimates
        self._update_ewma(trade)
        
        # Keep only recent trades
        if len(self._trades) > self.window_size:
            self._trades = self._trades[-self.window_size:]
    
    def estimate(self, setup: str, regime: Optional[str] = None) -> Tuple[float, float, float]:
        """Estimate win probability and average returns.
        
        Args:
            setup: Setup name
            regime: Regime name (if None, uses current regime)
            
        Returns:
            Tuple of (win_probability, avg_win_return, avg_loss_return)
        """
        if regime is None:
            regime = self._current_regime
        
        # Filter trades by setup and regime
        relevant_trades = self._filter_trades(setup, regime)
        
        if len(relevant_trades) < self.min_samples:
            # Use prior estimates
            return self._get_prior_estimates()
        
        # Calculate empirical estimates
        wins = [t for t in relevant_trades if t.is_winner]
        losses = [t for t in relevant_trades if t.is_loser]
        
        if not wins or not losses:
            return self._get_prior_estimates()
        
        # Win probability with Beta-Binomial smoothing
        n_wins = len(wins)
        n_losses = len(losses)
        n_total = n_wins + n_losses
        
        # Beta-Binomial posterior
        posterior_alpha = self.beta_prior_alpha + n_wins
        posterior_beta = self.beta_prior_beta + n_losses
        
        win_prob = posterior_alpha / (posterior_alpha + posterior_beta)
        
        # Average returns
        avg_win = np.mean([t.returns for t in wins])
        avg_loss = abs(np.mean([t.returns for t in losses]))
        
        # Apply EWMA smoothing if available
        if self._ewma_win_rate is not None:
            win_prob = self.alpha * win_prob + (1 - self.alpha) * self._ewma_win_rate
        
        if self._ewma_avg_win is not None:
            avg_win = self.alpha * avg_win + (1 - self.alpha) * self._ewma_avg_win
        
        if self._ewma_avg_loss is not None:
            avg_loss = self.alpha * avg_loss + (1 - self.alpha) * self._ewma_avg_loss
        
        return win_prob, avg_win, avg_loss
    
    def get_regime_probabilities(self) -> Dict[str, float]:
        """Get current regime probabilities.
        
        Returns:
            Dictionary mapping regime names to probabilities
        """
        return self._regime_probabilities.copy()
    
    def get_current_regime(self) -> str:
        """Get current regime.
        
        Returns:
            Current regime name
        """
        return self._current_regime
    
    def get_trade_count(self, setup: str, regime: Optional[str] = None) -> int:
        """Get number of trades for setup and regime.
        
        Args:
            setup: Setup name
            regime: Regime name (if None, uses current regime)
            
        Returns:
            Number of trades
        """
        if regime is None:
            regime = self._current_regime
        
        relevant_trades = self._filter_trades(setup, regime)
        return len(relevant_trades)
    
    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self._trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'current_regime': self._current_regime,
                'regime_probabilities': self._regime_probabilities,
            }
        
        wins = [t for t in self._trades if t.is_winner]
        losses = [t for t in self._trades if t.is_loser]
        
        win_rate = len(wins) / len(self._trades) if self._trades else 0.0
        avg_win = np.mean([t.returns for t in wins]) if wins else 0.0
        avg_loss = abs(np.mean([t.returns for t in losses])) if losses else 0.0
        
        # Profit factor
        total_wins = sum(t.returns for t in wins) if wins else 0.0
        total_losses = abs(sum(t.returns for t in losses)) if losses else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        return {
            'total_trades': len(self._trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'current_regime': self._current_regime,
            'regime_probabilities': self._regime_probabilities,
        }
    
    def _filter_trades(self, setup: str, regime: str) -> List[TradeResult]:
        """Filter trades by setup and regime.
        
        Args:
            setup: Setup name
            regime: Regime name
            
        Returns:
            Filtered list of trades
        """
        return [
            t for t in self._trades
            if t.setup == setup and self._get_trade_regime(t) == regime
        ]
    
    def _get_trade_regime(self, trade: TradeResult) -> str:
        """Get regime for a trade.
        
        Args:
            trade: Trade result
            
        Returns:
            Regime name
        """
        if not self.use_regimes:
            return "normal"
        
        # Find regime at trade time
        trade_time = trade.entry_timestamp
        
        # Simple regime assignment based on trade time
        # In practice, this would be more sophisticated
        return self._current_regime
    
    def _update_regime(self, trade: TradeResult) -> None:
        """Update regime based on trade.
        
        Args:
            trade: Trade result
        """
        if not self.use_regimes:
            return
        
        # Simple regime detection based on recent performance
        if len(self._trades) < self.regime_window:
            return
        
        recent_trades = self._trades[-self.regime_window:]
        recent_returns = [t.returns for t in recent_trades]
        
        if not recent_returns:
            return
        
        # Calculate regime metrics
        avg_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)
        
        # Regime classification
        if avg_return > self.regime_threshold and volatility < 0.02:
            new_regime = "bull"
        elif avg_return < -self.regime_threshold and volatility < 0.02:
            new_regime = "bear"
        elif volatility > 0.03:
            new_regime = "volatile"
        else:
            new_regime = "normal"
        
        # Update regime if changed
        if new_regime != self._current_regime:
            self._current_regime = new_regime
        
        # Update regime probabilities
        self._regime_probabilities = self._calculate_regime_probabilities()
    
    def _calculate_regime_probabilities(self) -> Dict[str, float]:
        """Calculate regime probabilities.
        
        Returns:
            Dictionary with regime probabilities
        """
        if not self._trades or len(self._trades) < self.regime_window:
            return {"normal": 1.0}
        
        # Count trades in each regime
        regime_counts = {"bull": 0, "bear": 0, "volatile": 0, "normal": 0}
        
        for trade in self._trades[-self.regime_window:]:
            regime = self._get_trade_regime(trade)
            regime_counts[regime] += 1
        
        total = sum(regime_counts.values())
        if total == 0:
            return {"normal": 1.0}
        
        # Convert to probabilities
        probabilities = {}
        for regime, count in regime_counts.items():
            probabilities[regime] = count / total
        
        return probabilities
    
    def _update_ewma(self, trade: TradeResult) -> None:
        """Update EWMA estimates.
        
        Args:
            trade: Trade result
        """
        if self._ewma_win_rate is None:
            self._ewma_win_rate = 1.0 if trade.is_winner else 0.0
        else:
            self._ewma_win_rate = (
                self.alpha * (1.0 if trade.is_winner else 0.0) +
                (1 - self.alpha) * self._ewma_win_rate
            )
        
        if trade.is_winner:
            if self._ewma_avg_win is None:
                self._ewma_avg_win = trade.returns
            else:
                self._ewma_avg_win = (
                    self.alpha * trade.returns +
                    (1 - self.alpha) * self._ewma_avg_win
                )
        else:
            if self._ewma_avg_loss is None:
                self._ewma_avg_loss = abs(trade.returns)
            else:
                self._ewma_avg_loss = (
                    self.alpha * abs(trade.returns) +
                    (1 - self.alpha) * self._ewma_avg_loss
                )
    
    def _get_prior_estimates(self) -> Tuple[float, float, float]:
        """Get prior estimates.
        
        Returns:
            Tuple of (win_probability, avg_win_return, avg_loss_return)
        """
        # Use Beta distribution prior
        win_prob = self.beta_prior_alpha / (self.beta_prior_alpha + self.beta_prior_beta)
        
        # Default returns (conservative estimates)
        avg_win = 0.02  # 2% average win
        avg_loss = 0.01  # 1% average loss
        
        return win_prob, avg_win, avg_loss

"""Risk guards and circuit breakers."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ...types import TradePlan, TradeResult


class RiskGuards:
    """Risk management guards and circuit breakers."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_size: float = 0.1,
        max_sector_exposure: float = 0.3,
        max_correlation: float = 0.7,
        daily_loss_limit: float = 0.05,
        weekly_loss_limit: float = 0.15,
        max_drawdown_limit: float = 0.25,
        max_positions: int = 20,
        min_positions: int = 1,
    ):
        """Initialize risk guards.
        
        Args:
            initial_capital: Initial portfolio capital
            max_position_size: Maximum position size as fraction of portfolio
            max_sector_exposure: Maximum sector exposure as fraction of portfolio
            max_correlation: Maximum correlation between positions
            daily_loss_limit: Daily loss limit as fraction of portfolio
            weekly_loss_limit: Weekly loss limit as fraction of portfolio
            max_drawdown_limit: Maximum drawdown limit as fraction of portfolio
            max_positions: Maximum number of positions
            min_positions: Minimum number of positions
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation = max_correlation
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.max_positions = max_positions
        self.min_positions = min_positions
        
        # Internal state
        self._current_capital = initial_capital
        self._peak_capital = initial_capital
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._current_positions: Dict[str, float] = {}
        self._position_history: List[TradeResult] = []
    
    def check_trade_plan(self, trade_plan: TradePlan) -> Dict[str, bool]:
        """Check if trade plan passes risk guards.
        
        Args:
            trade_plan: Trade plan to check
            
        Returns:
            Dictionary with guard check results
        """
        checks = {}
        
        # Check position size limit
        position_percentage = trade_plan.position_size / self._current_capital
        checks['position_size_ok'] = position_percentage <= self.max_position_size
        
        # Check maximum positions limit
        checks['max_positions_ok'] = len(self._current_positions) < self.max_positions
        
        # Check minimum positions (if closing positions)
        if trade_plan.signal.symbol in self._current_positions:
            remaining_positions = len(self._current_positions) - 1
            checks['min_positions_ok'] = remaining_positions >= self.min_positions
        else:
            checks['min_positions_ok'] = True
        
        # Check sector exposure (placeholder - would need sector data)
        checks['sector_exposure_ok'] = True
        
        # Check correlation (placeholder - would need correlation data)
        checks['correlation_ok'] = True
        
        # Check daily loss limit
        checks['daily_loss_ok'] = abs(self._daily_pnl) <= self._current_capital * self.daily_loss_limit
        
        # Check weekly loss limit
        checks['weekly_loss_ok'] = abs(self._weekly_pnl) <= self._current_capital * self.weekly_loss_limit
        
        # Check maximum drawdown
        current_drawdown = (self._peak_capital - self._current_capital) / self._peak_capital
        checks['max_drawdown_ok'] = current_drawdown <= self.max_drawdown_limit
        
        # Overall check
        checks['all_checks_passed'] = all(checks.values())
        
        return checks
    
    def update_position(self, symbol: str, size: float) -> None:
        """Update position size.
        
        Args:
            symbol: Stock symbol
            size: Position size
        """
        if size == 0:
            self._current_positions.pop(symbol, None)
        else:
            self._current_positions[symbol] = size
    
    def update_pnl(self, pnl: float, timestamp: pd.Timestamp) -> None:
        """Update P&L and check for circuit breakers.
        
        Args:
            pnl: P&L amount
            timestamp: Timestamp
        """
        self._current_capital += pnl
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        
        # Update peak capital
        if self._current_capital > self._peak_capital:
            self._peak_capital = self._current_capital
        
        # Reset daily P&L at start of new day
        if timestamp.date() != getattr(self, '_last_date', None):
            self._daily_pnl = 0.0
            self._last_date = timestamp.date()
        
        # Reset weekly P&L at start of new week
        if timestamp.week != getattr(self, '_last_week', None):
            self._weekly_pnl = 0.0
            self._last_week = timestamp.week
    
    def add_trade_result(self, trade_result: TradeResult) -> None:
        """Add trade result to history.
        
        Args:
            trade_result: Trade result
        """
        self._position_history.append(trade_result)
        self.update_pnl(trade_result.pnl, trade_result.exit_timestamp)
    
    def check_circuit_breakers(self) -> Dict[str, bool]:
        """Check circuit breakers.
        
        Returns:
            Dictionary with circuit breaker status
        """
        breakers = {}
        
        # Daily loss circuit breaker
        daily_loss_pct = abs(self._daily_pnl) / self._current_capital
        breakers['daily_loss_breaker'] = daily_loss_pct > self.daily_loss_limit
        
        # Weekly loss circuit breaker
        weekly_loss_pct = abs(self._weekly_pnl) / self._current_capital
        breakers['weekly_loss_breaker'] = weekly_loss_pct > self.weekly_loss_limit
        
        # Maximum drawdown circuit breaker
        current_drawdown = (self._peak_capital - self._current_capital) / self._peak_capital
        breakers['max_drawdown_breaker'] = current_drawdown > self.max_drawdown_limit
        
        # Overall circuit breaker
        breakers['any_breaker_active'] = any(breakers.values())
        
        return breakers
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        current_drawdown = (self._peak_capital - self._current_capital) / self._peak_capital
        
        return {
            'current_capital': self._current_capital,
            'peak_capital': self._peak_capital,
            'current_drawdown': current_drawdown,
            'daily_pnl': self._daily_pnl,
            'weekly_pnl': self._weekly_pnl,
            'daily_pnl_pct': self._daily_pnl / self._current_capital,
            'weekly_pnl_pct': self._weekly_pnl / self._current_capital,
            'num_positions': len(self._current_positions),
            'total_exposure': sum(self._current_positions.values()),
            'exposure_pct': sum(self._current_positions.values()) / self._current_capital,
        }
    
    def get_position_summary(self) -> Dict[str, float]:
        """Get position summary.
        
        Returns:
            Dictionary with position summary
        """
        if not self._current_positions:
            return {
                'num_positions': 0,
                'total_exposure': 0.0,
                'avg_position_size': 0.0,
                'max_position_size': 0.0,
                'min_position_size': 0.0,
            }
        
        position_sizes = list(self._current_positions.values())
        
        return {
            'num_positions': len(self._current_positions),
            'total_exposure': sum(position_sizes),
            'avg_position_size': sum(position_sizes) / len(position_sizes),
            'max_position_size': max(position_sizes),
            'min_position_size': min(position_sizes),
        }
    
    def get_trade_summary(self) -> Dict[str, float]:
        """Get trade summary.
        
        Returns:
            Dictionary with trade summary
        """
        if not self._position_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
            }
        
        wins = [t for t in self._position_history if t.is_winner]
        losses = [t for t in self._position_history if t.is_loser]
        
        return {
            'total_trades': len(self._position_history),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(self._position_history),
            'total_pnl': sum(t.pnl for t in self._position_history),
            'avg_win': sum(t.pnl for t in wins) / len(wins) if wins else 0.0,
            'avg_loss': sum(t.pnl for t in losses) / len(losses) if losses else 0.0,
        }
    
    def reset_daily(self) -> None:
        """Reset daily metrics."""
        self._daily_pnl = 0.0
    
    def reset_weekly(self) -> None:
        """Reset weekly metrics."""
        self._weekly_pnl = 0.0
    
    def reset_all(self) -> None:
        """Reset all metrics."""
        self._current_capital = self.initial_capital
        self._peak_capital = self.initial_capital
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._current_positions.clear()
        self._position_history.clear()
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get risk limits.
        
        Returns:
            Dictionary with risk limits
        """
        return {
            'max_position_size': self.max_position_size,
            'max_sector_exposure': self.max_sector_exposure,
            'max_correlation': self.max_correlation,
            'daily_loss_limit': self.daily_loss_limit,
            'weekly_loss_limit': self.weekly_loss_limit,
            'max_drawdown_limit': self.max_drawdown_limit,
            'max_positions': self.max_positions,
            'min_positions': self.min_positions,
        }

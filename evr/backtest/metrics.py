"""Backtest metrics calculation."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ...types import Metrics, TradeResult


class BacktestMetrics:
    """Backtest metrics calculator."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def calculate_metrics(
        self,
        trade_results: List[TradeResult],
        equity_curve: List[float],
        timestamps: List[pd.Timestamp],
        initial_capital: float,
    ) -> Metrics:
        """Calculate comprehensive backtest metrics.
        
        Args:
            trade_results: List of trade results
            equity_curve: Equity curve values
            timestamps: Timestamps for equity curve
            initial_capital: Initial capital
            
        Returns:
            Metrics object
        """
        if not trade_results:
            return self._create_empty_metrics()
        
        # Basic trade metrics
        total_trades = len(trade_results)
        winning_trades = [t for t in trade_results if t.is_winner]
        losing_trades = [t for t in trade_results if t.is_loser]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean([t.returns for t in winning_trades]) if winning_trades else 0.0
        avg_loss = abs(np.mean([t.returns for t in losing_trades])) if losing_trades else 0.0
        
        # Portfolio metrics
        final_capital = equity_curve[-1] if equity_curve else initial_capital
        total_return = (final_capital - initial_capital) / initial_capital
        
        # CAGR calculation
        if timestamps and len(timestamps) > 1:
            start_date = timestamps[0]
            end_date = timestamps[-1]
            years = (end_date - start_date).total_seconds() / (365.25 * 24 * 3600)
            cagr = (final_capital / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0
        else:
            cagr = 0.0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve, timestamps)
        sortino_ratio = self._calculate_sortino_ratio(equity_curve, timestamps)
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Trade-specific metrics
        largest_win = max([t.returns for t in winning_trades]) if winning_trades else 0.0
        largest_loss = min([t.returns for t in losing_trades]) if losing_trades else 0.0
        
        profit_factor = self._calculate_profit_factor(trade_results)
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        # Risk metrics
        var_95, expected_shortfall = self._calculate_risk_metrics(trade_results)
        
        # Consecutive trades
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_trades(trade_results)
        
        # Additional metrics
        turnover = self._calculate_turnover(trade_results, initial_capital)
        avg_trade_duration = np.mean([t.duration_days for t in trade_results]) if trade_results else 0.0
        
        # Monthly returns
        best_month, worst_month = self._calculate_monthly_returns(equity_curve, timestamps)
        
        return Metrics(
            total_return=total_return,
            cagr=cagr,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            max_consecutive_losses=max_consecutive_losses,
            max_consecutive_wins=max_consecutive_wins,
            turnover=turnover,
            avg_trade_duration=avg_trade_duration,
            best_month=best_month,
            worst_month=worst_month,
            start_date=timestamps[0] if timestamps else pd.Timestamp.now(),
            end_date=timestamps[-1] if timestamps else pd.Timestamp.now(),
        )
    
    def _create_empty_metrics(self) -> Metrics:
        """Create empty metrics.
        
        Returns:
            Empty metrics object
        """
        return Metrics(
            total_return=0.0,
            cagr=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            var_95=0.0,
            expected_shortfall=0.0,
            max_consecutive_losses=0,
            max_consecutive_wins=0,
            turnover=0.0,
            avg_trade_duration=0.0,
            best_month=0.0,
            worst_month=0.0,
            start_date=pd.Timestamp.now(),
            end_date=pd.Timestamp.now(),
        )
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown.
        
        Args:
            equity_curve: Equity curve values
            
        Returns:
            Maximum drawdown
        """
        if not equity_curve:
            return 0.0
        
        equity = pd.Series(equity_curve)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_sharpe_ratio(self, equity_curve: List[float], timestamps: List[pd.Timestamp]) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            equity_curve: Equity curve values
            timestamps: Timestamps
            
        Returns:
            Sharpe ratio
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        if returns.std() == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, equity_curve: List[float], timestamps: List[pd.Timestamp]) -> float:
        """Calculate Sortino ratio.
        
        Args:
            equity_curve: Equity curve values
            timestamps: Timestamps
            
        Returns:
            Sortino ratio
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        returns = pd.Series(equity_curve).pct_change().dropna()
        negative_returns = returns[returns < 0]
        
        if negative_returns.std() == 0:
            return 0.0
        
        # Annualized Sortino ratio
        return returns.mean() / negative_returns.std() * np.sqrt(252)
    
    def _calculate_profit_factor(self, trade_results: List[TradeResult]) -> float:
        """Calculate profit factor.
        
        Args:
            trade_results: List of trade results
            
        Returns:
            Profit factor
        """
        if not trade_results:
            return 0.0
        
        total_wins = sum(t.pnl for t in trade_results if t.is_winner)
        total_losses = abs(sum(t.pnl for t in trade_results if t.is_loser))
        
        return total_wins / total_losses if total_losses > 0 else 0.0
    
    def _calculate_risk_metrics(self, trade_results: List[TradeResult]) -> tuple[float, float]:
        """Calculate VaR and Expected Shortfall.
        
        Args:
            trade_results: List of trade results
            
        Returns:
            Tuple of (VaR_95, Expected_Shortfall)
        """
        if not trade_results:
            return 0.0, 0.0
        
        returns = [t.returns for t in trade_results]
        returns_series = pd.Series(returns)
        
        var_95 = returns_series.quantile(0.05)
        expected_shortfall = returns_series[returns_series <= var_95].mean()
        
        return abs(var_95), abs(expected_shortfall)
    
    def _calculate_consecutive_trades(self, trade_results: List[TradeResult]) -> tuple[int, int]:
        """Calculate maximum consecutive wins and losses.
        
        Args:
            trade_results: List of trade results
            
        Returns:
            Tuple of (max_consecutive_wins, max_consecutive_losses)
        """
        if not trade_results:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trade_results:
            if trade.is_winner:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _calculate_turnover(self, trade_results: List[TradeResult], initial_capital: float) -> float:
        """Calculate turnover.
        
        Args:
            trade_results: List of trade results
            initial_capital: Initial capital
            
        Returns:
            Turnover
        """
        if not trade_results or initial_capital <= 0:
            return 0.0
        
        total_volume = sum(abs(t.pnl) for t in trade_results)
        return total_volume / initial_capital
    
    def _calculate_monthly_returns(self, equity_curve: List[float], timestamps: List[pd.Timestamp]) -> tuple[float, float]:
        """Calculate best and worst monthly returns.
        
        Args:
            equity_curve: Equity curve values
            timestamps: Timestamps
            
        Returns:
            Tuple of (best_month, worst_month)
        """
        if not equity_curve or not timestamps:
            return 0.0, 0.0
        
        # Group by month and calculate returns
        df = pd.DataFrame({
            'equity': equity_curve,
            'timestamp': timestamps
        })
        df.set_index('timestamp', inplace=True)
        
        monthly_returns = df['equity'].resample('M').last().pct_change().dropna()
        
        if monthly_returns.empty:
            return 0.0, 0.0
        
        return monthly_returns.max(), monthly_returns.min()
    
    def calculate_setup_metrics(self, trade_results: List[TradeResult]) -> dict:
        """Calculate metrics by setup.
        
        Args:
            trade_results: List of trade results
            
        Returns:
            Dictionary with metrics by setup
        """
        if not trade_results:
            return {}
        
        # Group by setup
        setup_groups = {}
        for trade in trade_results:
            setup = trade.setup
            if setup not in setup_groups:
                setup_groups[setup] = []
            setup_groups[setup].append(trade)
        
        # Calculate metrics for each setup
        setup_metrics = {}
        for setup, trades in setup_groups.items():
            setup_metrics[setup] = {
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if t.is_winner]),
                'losing_trades': len([t for t in trades if t.is_loser]),
                'win_rate': len([t for t in trades if t.is_winner]) / len(trades),
                'avg_win': np.mean([t.returns for t in trades if t.is_winner]) if any(t.is_winner for t in trades) else 0.0,
                'avg_loss': abs(np.mean([t.returns for t in trades if t.is_loser])) if any(t.is_loser for t in trades) else 0.0,
                'total_pnl': sum(t.pnl for t in trades),
                'profit_factor': self._calculate_profit_factor(trades),
                'expectancy': np.mean([t.returns for t in trades]),
            }
        
        return setup_metrics
    
    def calculate_symbol_metrics(self, trade_results: List[TradeResult]) -> dict:
        """Calculate metrics by symbol.
        
        Args:
            trade_results: List of trade results
            
        Returns:
            Dictionary with metrics by symbol
        """
        if not trade_results:
            return {}
        
        # Group by symbol
        symbol_groups = {}
        for trade in trade_results:
            symbol = trade.symbol
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(trade)
        
        # Calculate metrics for each symbol
        symbol_metrics = {}
        for symbol, trades in symbol_groups.items():
            symbol_metrics[symbol] = {
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if t.is_winner]),
                'losing_trades': len([t for t in trades if t.is_loser]),
                'win_rate': len([t for t in trades if t.is_winner]) / len(trades),
                'avg_win': np.mean([t.returns for t in trades if t.is_winner]) if any(t.is_winner for t in trades) else 0.0,
                'avg_loss': abs(np.mean([t.returns for t in trades if t.is_loser])) if any(t.is_loser for t in trades) else 0.0,
                'total_pnl': sum(t.pnl for t in trades),
                'profit_factor': self._calculate_profit_factor(trades),
                'expectancy': np.mean([t.returns for t in trades]),
            }
        
        return symbol_metrics

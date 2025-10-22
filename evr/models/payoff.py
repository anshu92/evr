"""Payoff models for EVR."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...types import Bars, TradePlan


class PayoffModel:
    """Model for calculating trade payoffs and costs."""
    
    def __init__(
        self,
        commission_per_trade: float = 1.0,
        slippage_bps: float = 5.0,
        slippage_atr_multiplier: float = 0.5,
    ):
        """Initialize PayoffModel.
        
        Args:
            commission_per_trade: Commission per trade in dollars
            slippage_bps: Slippage in basis points
            slippage_atr_multiplier: ATR multiplier for slippage calculation
        """
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.slippage_atr_multiplier = slippage_atr_multiplier
    
    def calculate_costs(
        self,
        trade_plan: TradePlan,
        bars: Bars,
        timestamp: pd.Timestamp,
    ) -> Dict[str, float]:
        """Calculate trading costs for a trade plan.
        
        Args:
            trade_plan: Trade plan
            bars: OHLCV data
            timestamp: Current timestamp
            
        Returns:
            Dictionary with cost breakdown
        """
        costs = {}
        
        # Commission costs
        costs['commission'] = self.commission_per_trade
        
        # Slippage costs
        slippage = self._calculate_slippage(trade_plan, bars, timestamp)
        costs['slippage'] = slippage
        
        # Total costs
        costs['total'] = costs['commission'] + costs['slippage']
        
        # Cost as percentage of trade value
        trade_value = trade_plan.position_size
        costs['cost_percentage'] = costs['total'] / trade_value if trade_value > 0 else 0.0
        
        return costs
    
    def calculate_payoff(
        self,
        trade_plan: TradePlan,
        bars: Bars,
        timestamp: pd.Timestamp,
        exit_price: float,
        exit_timestamp: pd.Timestamp,
    ) -> Dict[str, float]:
        """Calculate trade payoff.
        
        Args:
            trade_plan: Trade plan
            bars: OHLCV data
            timestamp: Entry timestamp
            exit_price: Exit price
            exit_timestamp: Exit timestamp
            
        Returns:
            Dictionary with payoff breakdown
        """
        payoff = {}
        
        # Calculate costs
        costs = self.calculate_costs(trade_plan, bars, timestamp)
        
        # Calculate gross P&L
        if trade_plan.signal.direction > 0:  # Long position
            gross_pnl = (exit_price - trade_plan.entry_price) * trade_plan.position_size
        else:  # Short position
            gross_pnl = (trade_plan.entry_price - exit_price) * trade_plan.position_size
        
        # Calculate net P&L
        net_pnl = gross_pnl - costs['total']
        
        # Calculate returns
        gross_return = gross_pnl / trade_plan.position_size
        net_return = net_pnl / trade_plan.position_size
        
        # Calculate duration
        duration = (exit_timestamp - timestamp).total_seconds() / (24 * 3600)  # days
        
        payoff.update({
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'gross_return': gross_return,
            'net_return': net_return,
            'duration_days': duration,
            'costs': costs,
        })
        
        return payoff
    
    def calculate_expected_payoff(
        self,
        trade_plan: TradePlan,
        bars: Bars,
        timestamp: pd.Timestamp,
        win_probability: float,
        avg_win_return: float,
        avg_loss_return: float,
    ) -> Dict[str, float]:
        """Calculate expected trade payoff.
        
        Args:
            trade_plan: Trade plan
            bars: OHLCV data
            timestamp: Current timestamp
            win_probability: Win probability
            avg_win_return: Average win return
            avg_loss_return: Average loss return
            
        Returns:
            Dictionary with expected payoff breakdown
        """
        expected_payoff = {}
        
        # Calculate costs
        costs = self.calculate_costs(trade_plan, bars, timestamp)
        
        # Calculate expected returns
        expected_gross_return = (
            win_probability * avg_win_return -
            (1 - win_probability) * avg_loss_return
        )
        
        expected_net_return = expected_gross_return - costs['cost_percentage']
        
        # Calculate expected P&L
        expected_gross_pnl = expected_gross_return * trade_plan.position_size
        expected_net_pnl = expected_net_return * trade_plan.position_size
        
        # Calculate risk-reward ratio
        risk_reward_ratio = trade_plan.risk_reward_ratio
        
        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(
            win_probability, avg_win_return, avg_loss_return
        )
        
        expected_payoff.update({
            'expected_gross_return': expected_gross_return,
            'expected_net_return': expected_net_return,
            'expected_gross_pnl': expected_gross_pnl,
            'expected_net_pnl': expected_net_pnl,
            'risk_reward_ratio': risk_reward_ratio,
            'kelly_fraction': kelly_fraction,
            'costs': costs,
        })
        
        return expected_payoff
    
    def _calculate_slippage(
        self,
        trade_plan: TradePlan,
        bars: Bars,
        timestamp: pd.Timestamp,
    ) -> float:
        """Calculate slippage cost.
        
        Args:
            trade_plan: Trade plan
            bars: OHLCV data
            timestamp: Current timestamp
            
        Returns:
            Slippage cost in dollars
        """
        try:
            # Get current bar
            current_bar = bars.loc[timestamp]
            
            # Calculate ATR-based slippage
            atr = self._get_atr(bars, timestamp)
            atr_slippage = atr * self.slippage_atr_multiplier
            
            # Calculate percentage-based slippage
            price_slippage = trade_plan.entry_price * (self.slippage_bps / 10000)
            
            # Use the larger of the two
            slippage_per_share = max(atr_slippage, price_slippage)
            
            # Calculate total slippage cost
            total_slippage = slippage_per_share * trade_plan.position_size
            
            return total_slippage
            
        except (KeyError, IndexError):
            # Fallback to percentage-based slippage
            return trade_plan.entry_price * trade_plan.position_size * (self.slippage_bps / 10000)
    
    def _get_atr(self, bars: Bars, timestamp: pd.Timestamp) -> float:
        """Get ATR value.
        
        Args:
            bars: OHLCV data
            timestamp: Current timestamp
            
        Returns:
            ATR value
        """
        try:
            # Calculate ATR manually
            high = bars['High']
            low = bars['Low']
            close = bars['Close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR as 14-period average
            atr = true_range.rolling(14).mean()
            
            return atr.loc[timestamp]
            
        except (KeyError, IndexError):
            # Fallback to simple range
            current_bar = bars.loc[timestamp]
            return current_bar['High'] - current_bar['Low']
    
    def _calculate_kelly_fraction(
        self,
        win_probability: float,
        avg_win_return: float,
        avg_loss_return: float,
    ) -> float:
        """Calculate Kelly fraction.
        
        Args:
            win_probability: Win probability
            avg_win_return: Average win return
            avg_loss_return: Average loss return
            
        Returns:
            Kelly fraction
        """
        if avg_loss_return <= 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win_return, p = win_probability, q = 1 - win_probability
        b = avg_win_return
        p = win_probability
        q = 1 - win_probability
        
        kelly = (b * p - q) / b
        
        # Ensure Kelly fraction is between 0 and 1
        return max(0.0, min(1.0, kelly))
    
    def get_cost_breakdown(self, trade_plan: TradePlan, bars: Bars, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Get detailed cost breakdown.
        
        Args:
            trade_plan: Trade plan
            bars: OHLCV data
            timestamp: Current timestamp
            
        Returns:
            Dictionary with detailed cost breakdown
        """
        costs = self.calculate_costs(trade_plan, bars, timestamp)
        
        # Add percentage breakdown
        total_costs = costs['total']
        if total_costs > 0:
            costs['commission_pct'] = costs['commission'] / total_costs
            costs['slippage_pct'] = costs['slippage'] / total_costs
        else:
            costs['commission_pct'] = 0.0
            costs['slippage_pct'] = 0.0
        
        return costs

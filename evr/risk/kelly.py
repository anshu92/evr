"""Kelly sizing for position management."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ...types import TradePlan


class KellySizing:
    """Kelly criterion-based position sizing."""
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_kelly_fraction: float = 0.5,
        max_position_size: float = 0.1,
        min_position_size: float = 0.01,
    ):
        """Initialize Kelly sizing.
        
        Args:
            kelly_fraction: Fraction of Kelly to use (0-1)
            max_kelly_fraction: Maximum Kelly fraction cap
            max_position_size: Maximum position size as fraction of portfolio
            min_position_size: Minimum position size as fraction of portfolio
        """
        self.kelly_fraction = kelly_fraction
        self.max_kelly_fraction = max_kelly_fraction
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
    
    def kelly_fraction(
        self,
        win_probability: float,
        avg_win_return: float,
        avg_loss_return: float,
    ) -> float:
        """Calculate Kelly fraction.
        
        Args:
            win_probability: Win probability (0-1)
            avg_win_return: Average win return ratio
            avg_loss_return: Average loss return ratio
            
        Returns:
            Kelly fraction (0-1)
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
        kelly = max(0.0, min(1.0, kelly))
        
        # Apply Kelly fraction multiplier
        kelly *= self.kelly_fraction
        
        # Apply maximum Kelly cap
        kelly = min(kelly, self.max_kelly_fraction)
        
        return kelly
    
    def size_position(
        self,
        trade_plan: TradePlan,
        portfolio_value: float,
        win_probability: float,
        avg_win_return: float,
        avg_loss_return: float,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Size position using Kelly criterion.
        
        Args:
            trade_plan: Trade plan
            portfolio_value: Current portfolio value
            win_probability: Win probability
            avg_win_return: Average win return
            avg_loss_return: Average loss return
            current_positions: Current positions (symbol -> size)
            
        Returns:
            Dictionary with sizing information
        """
        # Calculate Kelly fraction
        kelly = self.kelly_fraction(win_probability, avg_win_return, avg_loss_return)
        
        # Calculate position size based on risk
        risk_amount = portfolio_value * trade_plan.risk_per_trade
        stop_distance = abs(trade_plan.entry_price - trade_plan.stop_loss)
        
        if stop_distance <= 0:
            position_size = 0.0
        else:
            position_size = risk_amount / stop_distance
        
        # Convert to dollar amount
        position_value = position_size * trade_plan.entry_price
        
        # Apply Kelly sizing
        kelly_position_value = position_value * kelly
        
        # Apply position size limits
        max_position_value = portfolio_value * self.max_position_size
        min_position_value = portfolio_value * self.min_position_size
        
        # Final position sizing
        final_position_value = max(
            min(kelly_position_value, max_position_value),
            min_position_value if kelly > 0 else 0.0
        )
        
        # Check for position concentration limits
        if current_positions:
            total_exposure = sum(current_positions.values())
            if total_exposure + final_position_value > portfolio_value * 0.8:  # 80% max exposure
                final_position_value = max(0.0, portfolio_value * 0.8 - total_exposure)
        
        # Calculate final position size
        final_position_size = final_position_value / trade_plan.entry_price if trade_plan.entry_price > 0 else 0.0
        
        return {
            'kelly_fraction': kelly,
            'position_size': final_position_size,
            'position_value': final_position_value,
            'risk_amount': risk_amount,
            'stop_distance': stop_distance,
            'position_percentage': final_position_value / portfolio_value,
            'kelly_position_value': kelly_position_value,
            'max_position_value': max_position_value,
            'min_position_value': min_position_value,
        }
    
    def calculate_optimal_size(
        self,
        entry_price: float,
        stop_loss: float,
        win_probability: float,
        avg_win_return: float,
        avg_loss_return: float,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """Calculate optimal position size.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            win_probability: Win probability
            avg_win_return: Average win return
            avg_loss_return: Average loss return
            portfolio_value: Portfolio value
            
        Returns:
            Dictionary with optimal sizing information
        """
        # Calculate Kelly fraction
        kelly = self.kelly_fraction(win_probability, avg_win_return, avg_loss_return)
        
        # Calculate risk per trade
        risk_per_trade = 0.02  # 2% default risk per trade
        
        # Calculate position size
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            return {
                'kelly_fraction': kelly,
                'position_size': 0.0,
                'position_value': 0.0,
                'position_percentage': 0.0,
                'risk_amount': 0.0,
            }
        
        risk_amount = portfolio_value * risk_per_trade
        position_size = risk_amount / stop_distance
        position_value = position_size * entry_price
        
        # Apply Kelly sizing
        kelly_position_value = position_value * kelly
        
        # Apply limits
        max_position_value = portfolio_value * self.max_position_size
        final_position_value = min(kelly_position_value, max_position_value)
        
        final_position_size = final_position_value / entry_price
        
        return {
            'kelly_fraction': kelly,
            'position_size': final_position_size,
            'position_value': final_position_value,
            'position_percentage': final_position_value / portfolio_value,
            'risk_amount': risk_amount,
            'stop_distance': stop_distance,
        }
    
    def validate_sizing(
        self,
        position_size: float,
        position_value: float,
        portfolio_value: float,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, bool]:
        """Validate position sizing.
        
        Args:
            position_size: Position size
            position_value: Position value
            portfolio_value: Portfolio value
            current_positions: Current positions
            
        Returns:
            Dictionary with validation results
        """
        validation = {}
        
        # Check position size limits
        position_percentage = position_value / portfolio_value
        validation['within_size_limits'] = (
            self.min_position_size <= position_percentage <= self.max_position_size
        )
        
        # Check for positive position
        validation['positive_position'] = position_size > 0
        
        # Check for sufficient capital
        validation['sufficient_capital'] = position_value <= portfolio_value
        
        # Check position concentration
        if current_positions:
            total_exposure = sum(current_positions.values())
            validation['within_concentration_limits'] = (
                total_exposure + position_value <= portfolio_value * 0.8
            )
        else:
            validation['within_concentration_limits'] = True
        
        # Overall validation
        validation['valid'] = all(validation.values())
        
        return validation
    
    def get_sizing_limits(self, portfolio_value: float) -> Dict[str, float]:
        """Get position sizing limits.
        
        Args:
            portfolio_value: Portfolio value
            
        Returns:
            Dictionary with sizing limits
        """
        return {
            'max_position_value': portfolio_value * self.max_position_size,
            'min_position_value': portfolio_value * self.min_position_size,
            'max_position_percentage': self.max_position_size,
            'min_position_percentage': self.min_position_size,
            'max_kelly_fraction': self.max_kelly_fraction,
            'kelly_fraction_multiplier': self.kelly_fraction,
        }

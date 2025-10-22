"""Squeeze Breakout setup."""

from __future__ import annotations

from typing import List

import pandas as pd

from ...types import Bars, Features, Signal
from ..base import BaseSetup


class SqueezeBreakout(BaseSetup):
    """Squeeze Breakout trading setup.
    
    This setup identifies periods of low volatility (squeeze) followed by
    breakouts in either direction. It uses Bollinger Bands and Keltner Channels
    to identify squeeze conditions.
    """
    
    def __init__(self, name: str = "squeeze_breakout", **kwargs):
        """Initialize Squeeze Breakout setup.
        
        Args:
            name: Setup name
            **kwargs: Setup parameters
        """
        super().__init__(name, **kwargs)
        
        # Default parameters
        self.bb_period = kwargs.get('bb_period', 20)
        self.bb_std = kwargs.get('bb_std', 2.0)
        self.kc_period = kwargs.get('kc_period', 20)
        self.kc_multiplier = kwargs.get('kc_multiplier', 1.5)
        self.min_squeeze_bars = kwargs.get('min_squeeze_bars', 5)
        self.breakout_threshold = kwargs.get('breakout_threshold', 0.02)
    
    def signals(
        self,
        bars: Bars,
        features: Features,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> List[Signal]:
        """Generate squeeze breakout signals.
        
        Args:
            bars: OHLCV data
            features: Computed features
            symbol: Stock symbol
            timestamp: Current timestamp
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if len(bars) < max(self.bb_period, self.kc_period) + self.min_squeeze_bars:
            return signals
        
        # Get current and previous values
        current_idx = bars.index.get_loc(timestamp)
        if current_idx < self.min_squeeze_bars:
            return signals
        
        # Check for squeeze condition
        is_squeeze = self._is_squeeze(bars, current_idx)
        
        if not is_squeeze:
            return signals
        
        # Check for breakout
        breakout_direction = self._check_breakout(bars, current_idx)
        
        if breakout_direction != 0:
            # Calculate signal strength
            strength = self._calculate_strength(bars, current_idx, breakout_direction)
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                timestamp=timestamp,
                direction=breakout_direction,
                strength=strength,
                setup=self.name,
                features=self._get_signal_features(bars, features, current_idx),
                metadata={
                    'squeeze_bars': self._count_squeeze_bars(bars, current_idx),
                    'breakout_type': 'upper' if breakout_direction > 0 else 'lower',
                }
            )
            
            signals.append(signal)
        
        return signals
    
    def _is_squeeze(self, bars: Bars, current_idx: int) -> bool:
        """Check if current bar is in squeeze condition.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            
        Returns:
            True if in squeeze condition
        """
        # Get Bollinger Bands and Keltner Channels
        bb_upper = bars['Close'].rolling(self.bb_period).mean() + \
                  bars['Close'].rolling(self.bb_period).std() * self.bb_std
        bb_lower = bars['Close'].rolling(self.bb_period).mean() - \
                  bars['Close'].rolling(self.bb_period).std() * self.bb_std
        
        # Keltner Channels (using ATR)
        atr = bars['High'].rolling(self.kc_period).max() - \
              bars['Low'].rolling(self.kc_period).min()
        kc_upper = bars['Close'].rolling(self.kc_period).mean() + \
                   atr * self.kc_multiplier
        kc_lower = bars['Close'].rolling(self.kc_period).mean() - \
                   atr * self.kc_multiplier
        
        # Check if BB is inside KC
        current_bb_upper = bb_upper.iloc[current_idx]
        current_bb_lower = bb_lower.iloc[current_idx]
        current_kc_upper = kc_upper.iloc[current_idx]
        current_kc_lower = kc_lower.iloc[current_idx]
        
        return (current_bb_upper < current_kc_upper and 
                current_bb_lower > current_kc_lower)
    
    def _check_breakout(self, bars: Bars, current_idx: int) -> int:
        """Check for breakout direction.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            
        Returns:
            1 for upward breakout, -1 for downward breakout, 0 for no breakout
        """
        if current_idx < 1:
            return 0
        
        current_bar = bars.iloc[current_idx]
        previous_bar = bars.iloc[current_idx - 1]
        
        # Calculate breakout threshold
        threshold = current_bar['Close'] * self.breakout_threshold
        
        # Check for upward breakout
        if (current_bar['Close'] > previous_bar['High'] + threshold and
            current_bar['Volume'] > bars['Volume'].rolling(20).mean().iloc[current_idx]):
            return 1
        
        # Check for downward breakout
        if (current_bar['Close'] < previous_bar['Low'] - threshold and
            current_bar['Volume'] > bars['Volume'].rolling(20).mean().iloc[current_idx]):
            return -1
        
        return 0
    
    def _calculate_strength(self, bars: Bars, current_idx: int, direction: int) -> float:
        """Calculate signal strength.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            direction: Breakout direction
            
        Returns:
            Signal strength (0-1)
        """
        # Base strength from volume
        current_volume = bars['Volume'].iloc[current_idx]
        avg_volume = bars['Volume'].rolling(20).mean().iloc[current_idx]
        volume_strength = min(current_volume / avg_volume, 3.0) / 3.0
        
        # Strength from price movement
        if current_idx < 1:
            return volume_strength
        
        previous_bar = bars.iloc[current_idx - 1]
        current_bar = bars.iloc[current_idx]
        
        if direction > 0:
            price_strength = (current_bar['Close'] - previous_bar['High']) / previous_bar['Close']
        else:
            price_strength = (previous_bar['Low'] - current_bar['Close']) / previous_bar['Close']
        
        price_strength = min(price_strength / self.breakout_threshold, 1.0)
        
        # Combine strengths
        return (volume_strength * 0.6 + price_strength * 0.4)
    
    def _count_squeeze_bars(self, bars: Bars, current_idx: int) -> int:
        """Count consecutive squeeze bars.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            
        Returns:
            Number of consecutive squeeze bars
        """
        count = 0
        for i in range(current_idx, max(0, current_idx - 20), -1):
            if self._is_squeeze(bars, i):
                count += 1
            else:
                break
        return count
    
    def _get_signal_features(self, bars: Bars, features: Features, current_idx: int) -> dict:
        """Get features for signal.
        
        Args:
            bars: OHLCV data
            features: Computed features
            current_idx: Current bar index
            
        Returns:
            Dictionary with signal features
        """
        feature_dict = {}
        
        # Add basic price features
        current_bar = bars.iloc[current_idx]
        feature_dict['close'] = current_bar['Close']
        feature_dict['volume'] = current_bar['Volume']
        feature_dict['high'] = current_bar['High']
        feature_dict['low'] = current_bar['Low']
        
        # Add technical indicators if available
        for col in features.columns:
            if current_idx < len(features):
                feature_dict[col] = features[col].iloc[current_idx]
        
        return feature_dict
    
    def validate_parameters(self) -> bool:
        """Validate setup parameters.
        
        Returns:
            True if parameters are valid
        """
        if self.bb_period <= 0 or self.kc_period <= 0:
            return False
        
        if self.bb_std <= 0 or self.kc_multiplier <= 0:
            return False
        
        if self.min_squeeze_bars < 1:
            return False
        
        if self.breakout_threshold <= 0:
            return False
        
        return True
    
    def get_required_features(self) -> List[str]:
        """Get list of required features for this setup.
        
        Returns:
            List of required feature names
        """
        return ['bb_width_20', 'atr_14', 'volume_ratio_20']
    
    def get_description(self) -> str:
        """Get setup description.
        
        Returns:
            Setup description
        """
        return (
            "Squeeze Breakout setup identifies periods of low volatility "
            "followed by breakouts. Uses Bollinger Bands and Keltner Channels "
            "to detect squeeze conditions."
        )

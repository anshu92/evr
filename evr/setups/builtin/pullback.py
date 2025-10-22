"""Trend Pullback setup."""

from __future__ import annotations

from typing import List

import pandas as pd

from ...types import Bars, Features, Signal
from ..base import BaseSetup


class TrendPullback(BaseSetup):
    """Trend Pullback trading setup.
    
    This setup identifies pullbacks in trending markets and looks for
    continuation signals. It uses moving averages and trend strength
    indicators to identify trends and pullback opportunities.
    """
    
    def __init__(self, name: str = "trend_pullback", **kwargs):
        """Initialize Trend Pullback setup.
        
        Args:
            name: Setup name
            **kwargs: Setup parameters
        """
        super().__init__(name, **kwargs)
        
        # Default parameters
        self.fast_ma = kwargs.get('fast_ma', 20)
        self.slow_ma = kwargs.get('slow_ma', 50)
        self.trend_strength_period = kwargs.get('trend_strength_period', 14)
        self.pullback_threshold = kwargs.get('pullback_threshold', 0.02)
        self.min_trend_bars = kwargs.get('min_trend_bars', 10)
        self.max_pullback_bars = kwargs.get('max_pullback_bars', 5)
    
    def signals(
        self,
        bars: Bars,
        features: Features,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> List[Signal]:
        """Generate trend pullback signals.
        
        Args:
            bars: OHLCV data
            features: Computed features
            symbol: Stock symbol
            timestamp: Current timestamp
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if len(bars) < max(self.fast_ma, self.slow_ma) + self.min_trend_bars:
            return signals
        
        # Get current and previous values
        current_idx = bars.index.get_loc(timestamp)
        if current_idx < self.min_trend_bars:
            return signals
        
        # Check for trend
        trend_direction = self._identify_trend(bars, current_idx)
        
        if trend_direction == 0:
            return signals
        
        # Check for pullback
        pullback_signal = self._check_pullback(bars, current_idx, trend_direction)
        
        if pullback_signal:
            # Calculate signal strength
            strength = self._calculate_strength(bars, current_idx, trend_direction)
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                timestamp=timestamp,
                direction=trend_direction,
                strength=strength,
                setup=self.name,
                features=self._get_signal_features(bars, features, current_idx),
                metadata={
                    'trend_direction': 'up' if trend_direction > 0 else 'down',
                    'pullback_type': pullback_signal['type'],
                    'trend_strength': pullback_signal['trend_strength'],
                }
            )
            
            signals.append(signal)
        
        return signals
    
    def _identify_trend(self, bars: Bars, current_idx: int) -> int:
        """Identify current trend direction.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            
        Returns:
            1 for uptrend, -1 for downtrend, 0 for no trend
        """
        # Calculate moving averages
        fast_ma = bars['Close'].rolling(self.fast_ma).mean()
        slow_ma = bars['Close'].rolling(self.slow_ma).mean()
        
        current_fast = fast_ma.iloc[current_idx]
        current_slow = slow_ma.iloc[current_idx]
        
        # Check if fast MA is above slow MA (uptrend)
        if current_fast > current_slow:
            # Verify trend strength
            if self._get_trend_strength(bars, current_idx) > 0.6:
                return 1
        
        # Check if fast MA is below slow MA (downtrend)
        elif current_fast < current_slow:
            # Verify trend strength
            if self._get_trend_strength(bars, current_idx) > 0.6:
                return -1
        
        return 0
    
    def _get_trend_strength(self, bars: Bars, current_idx: int) -> float:
        """Calculate trend strength.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            
        Returns:
            Trend strength (0-1)
        """
        if current_idx < self.trend_strength_period:
            return 0.0
        
        # Calculate price change over trend period
        start_price = bars['Close'].iloc[current_idx - self.trend_strength_period]
        end_price = bars['Close'].iloc[current_idx]
        
        price_change = abs(end_price - start_price) / start_price
        
        # Calculate consistency of trend
        fast_ma = bars['Close'].rolling(self.fast_ma).mean()
        slow_ma = bars['Close'].rolling(self.slow_ma).mean()
        
        # Count how many bars the fast MA was above/below slow MA
        trend_bars = 0
        for i in range(current_idx - self.trend_strength_period + 1, current_idx + 1):
            if fast_ma.iloc[i] > slow_ma.iloc[i]:
                trend_bars += 1
            elif fast_ma.iloc[i] < slow_ma.iloc[i]:
                trend_bars -= 1
        
        consistency = abs(trend_bars) / self.trend_strength_period
        
        # Combine price change and consistency
        return min(price_change * 2 + consistency, 1.0)
    
    def _check_pullback(self, bars: Bars, current_idx: int, trend_direction: int) -> Optional[dict]:
        """Check for pullback condition.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            trend_direction: Trend direction
            
        Returns:
            Pullback signal info or None
        """
        if current_idx < self.max_pullback_bars:
            return None
        
        # Look for pullback in the last few bars
        pullback_found = False
        pullback_type = None
        
        for i in range(current_idx - self.max_pullback_bars, current_idx):
            if self._is_pullback_bar(bars, i, trend_direction):
                pullback_found = True
                pullback_type = self._classify_pullback(bars, i, trend_direction)
                break
        
        if not pullback_found:
            return None
        
        # Check for reversal signal
        if self._is_reversal_signal(bars, current_idx, trend_direction):
            return {
                'type': pullback_type,
                'trend_strength': self._get_trend_strength(bars, current_idx),
            }
        
        return None
    
    def _is_pullback_bar(self, bars: Bars, idx: int, trend_direction: int) -> bool:
        """Check if bar represents a pullback.
        
        Args:
            bars: OHLCV data
            idx: Bar index
            trend_direction: Trend direction
            
        Returns:
            True if bar is a pullback
        """
        if idx < 1:
            return False
        
        current_bar = bars.iloc[idx]
        previous_bar = bars.iloc[idx - 1]
        
        if trend_direction > 0:  # Uptrend
            # Look for downward movement
            return current_bar['Close'] < previous_bar['Close'] * (1 - self.pullback_threshold)
        else:  # Downtrend
            # Look for upward movement
            return current_bar['Close'] > previous_bar['Close'] * (1 + self.pullback_threshold)
    
    def _classify_pullback(self, bars: Bars, idx: int, trend_direction: int) -> str:
        """Classify pullback type.
        
        Args:
            bars: OHLCV data
            idx: Bar index
            trend_direction: Trend direction
            
        Returns:
            Pullback type
        """
        if trend_direction > 0:
            return "bullish_pullback"
        else:
            return "bearish_pullback"
    
    def _is_reversal_signal(self, bars: Bars, current_idx: int, trend_direction: int) -> bool:
        """Check for reversal signal after pullback.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            trend_direction: Trend direction
            
        Returns:
            True if reversal signal detected
        """
        if current_idx < 1:
            return False
        
        current_bar = bars.iloc[current_idx]
        previous_bar = bars.iloc[current_idx - 1]
        
        if trend_direction > 0:  # Uptrend
            # Look for upward reversal
            return (current_bar['Close'] > previous_bar['Close'] and
                    current_bar['Volume'] > bars['Volume'].rolling(20).mean().iloc[current_idx])
        else:  # Downtrend
            # Look for downward reversal
            return (current_bar['Close'] < previous_bar['Close'] and
                    current_bar['Volume'] > bars['Volume'].rolling(20).mean().iloc[current_idx])
    
    def _calculate_strength(self, bars: Bars, current_idx: int, trend_direction: int) -> float:
        """Calculate signal strength.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            trend_direction: Trend direction
            
        Returns:
            Signal strength (0-1)
        """
        # Base strength from trend strength
        trend_strength = self._get_trend_strength(bars, current_idx)
        
        # Volume confirmation
        current_volume = bars['Volume'].iloc[current_idx]
        avg_volume = bars['Volume'].rolling(20).mean().iloc[current_idx]
        volume_strength = min(current_volume / avg_volume, 2.0) / 2.0
        
        # Price action strength
        if current_idx < 1:
            return trend_strength * 0.7 + volume_strength * 0.3
        
        previous_bar = bars.iloc[current_idx - 1]
        current_bar = bars.iloc[current_idx]
        
        if trend_direction > 0:
            price_strength = (current_bar['Close'] - previous_bar['Close']) / previous_bar['Close']
        else:
            price_strength = (previous_bar['Close'] - current_bar['Close']) / previous_bar['Close']
        
        price_strength = min(max(price_strength, 0), 0.05) / 0.05
        
        # Combine strengths
        return (trend_strength * 0.5 + volume_strength * 0.3 + price_strength * 0.2)
    
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
        
        # Add moving averages
        fast_ma = bars['Close'].rolling(self.fast_ma).mean()
        slow_ma = bars['Close'].rolling(self.slow_ma).mean()
        feature_dict['fast_ma'] = fast_ma.iloc[current_idx]
        feature_dict['slow_ma'] = slow_ma.iloc[current_idx]
        
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
        if self.fast_ma <= 0 or self.slow_ma <= 0:
            return False
        
        if self.fast_ma >= self.slow_ma:
            return False
        
        if self.trend_strength_period <= 0:
            return False
        
        if self.pullback_threshold <= 0:
            return False
        
        if self.min_trend_bars < 1 or self.max_pullback_bars < 1:
            return False
        
        return True
    
    def get_required_features(self) -> List[str]:
        """Get list of required features for this setup.
        
        Returns:
            List of required feature names
        """
        return ['sma_20', 'sma_50', 'volume_ratio_20', 'adx_14']
    
    def get_description(self) -> str:
        """Get setup description.
        
        Returns:
            Setup description
        """
        return (
            "Trend Pullback setup identifies pullbacks in trending markets "
            "and looks for continuation signals. Uses moving averages and "
            "trend strength indicators to identify trends and pullback opportunities."
        )

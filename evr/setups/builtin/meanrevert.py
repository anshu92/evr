"""Mean Reversion setup."""

from __future__ import annotations

from typing import List

import pandas as pd

from ...types import Bars, Features, Signal
from ..base import BaseSetup


class MeanReversion(BaseSetup):
    """Mean Reversion trading setup.
    
    This setup identifies overbought/oversold conditions and looks for
    mean reversion opportunities. It uses RSI, Bollinger Bands, and
    Z-score indicators to identify extreme price levels.
    """
    
    def __init__(self, name: str = "mean_reversion", **kwargs):
        """Initialize Mean Reversion setup.
        
        Args:
            name: Setup name
            **kwargs: Setup parameters
        """
        super().__init__(name, **kwargs)
        
        # Default parameters
        self.rsi_period = kwargs.get('rsi_period', 14)
        self.rsi_oversold = kwargs.get('rsi_oversold', 30)
        self.rsi_overbought = kwargs.get('rsi_overbought', 70)
        self.bb_period = kwargs.get('bb_period', 20)
        self.bb_std = kwargs.get('bb_std', 2.0)
        self.zscore_period = kwargs.get('zscore_period', 20)
        self.zscore_threshold = kwargs.get('zscore_threshold', 2.0)
        self.min_reversion_bars = kwargs.get('min_reversion_bars', 3)
    
    def signals(
        self,
        bars: Bars,
        features: Features,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> List[Signal]:
        """Generate mean reversion signals.
        
        Args:
            bars: OHLCV data
            features: Computed features
            symbol: Stock symbol
            timestamp: Current timestamp
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if len(bars) < max(self.rsi_period, self.bb_period, self.zscore_period) + self.min_reversion_bars:
            return signals
        
        # Get current and previous values
        current_idx = bars.index.get_loc(timestamp)
        if current_idx < self.min_reversion_bars:
            return signals
        
        # Check for oversold condition (buy signal)
        if self._is_oversold(bars, current_idx):
            strength = self._calculate_strength(bars, current_idx, 1)
            
            signal = Signal(
                symbol=symbol,
                timestamp=timestamp,
                direction=1,
                strength=strength,
                setup=self.name,
                features=self._get_signal_features(bars, features, current_idx),
                metadata={
                    'signal_type': 'oversold_buy',
                    'rsi_value': self._get_rsi(bars, current_idx),
                    'zscore_value': self._get_zscore(bars, current_idx),
                }
            )
            
            signals.append(signal)
        
        # Check for overbought condition (sell signal)
        elif self._is_overbought(bars, current_idx):
            strength = self._calculate_strength(bars, current_idx, -1)
            
            signal = Signal(
                symbol=symbol,
                timestamp=timestamp,
                direction=-1,
                strength=strength,
                setup=self.name,
                features=self._get_signal_features(bars, features, current_idx),
                metadata={
                    'signal_type': 'overbought_sell',
                    'rsi_value': self._get_rsi(bars, current_idx),
                    'zscore_value': self._get_zscore(bars, current_idx),
                }
            )
            
            signals.append(signal)
        
        return signals
    
    def _is_oversold(self, bars: Bars, current_idx: int) -> bool:
        """Check for oversold condition.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            
        Returns:
            True if oversold condition detected
        """
        # Check RSI
        rsi = self._get_rsi(bars, current_idx)
        if rsi is None or rsi > self.rsi_oversold:
            return False
        
        # Check Z-score
        zscore = self._get_zscore(bars, current_idx)
        if zscore is None or zscore > -self.zscore_threshold:
            return False
        
        # Check Bollinger Bands
        bb_position = self._get_bb_position(bars, current_idx)
        if bb_position is None or bb_position > 0.2:  # Not near lower band
            return False
        
        # Check for confirmation in previous bars
        return self._has_reversion_confirmation(bars, current_idx, 1)
    
    def _is_overbought(self, bars: Bars, current_idx: int) -> bool:
        """Check for overbought condition.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            
        Returns:
            True if overbought condition detected
        """
        # Check RSI
        rsi = self._get_rsi(bars, current_idx)
        if rsi is None or rsi < self.rsi_overbought:
            return False
        
        # Check Z-score
        zscore = self._get_zscore(bars, current_idx)
        if zscore is None or zscore < self.zscore_threshold:
            return False
        
        # Check Bollinger Bands
        bb_position = self._get_bb_position(bars, current_idx)
        if bb_position is None or bb_position < 0.8:  # Not near upper band
            return False
        
        # Check for confirmation in previous bars
        return self._has_reversion_confirmation(bars, current_idx, -1)
    
    def _get_rsi(self, bars: Bars, current_idx: int) -> Optional[float]:
        """Get RSI value.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            
        Returns:
            RSI value or None
        """
        if current_idx < self.rsi_period:
            return None
        
        # Calculate RSI manually
        prices = bars['Close'].iloc[current_idx - self.rsi_period:current_idx + 1]
        deltas = prices.diff()
        
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _get_zscore(self, bars: Bars, current_idx: int) -> Optional[float]:
        """Get Z-score value.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            
        Returns:
            Z-score value or None
        """
        if current_idx < self.zscore_period:
            return None
        
        prices = bars['Close'].iloc[current_idx - self.zscore_period:current_idx + 1]
        mean = prices.mean()
        std = prices.std()
        
        if std == 0:
            return 0.0
        
        return (prices.iloc[-1] - mean) / std
    
    def _get_bb_position(self, bars: Bars, current_idx: int) -> Optional[float]:
        """Get Bollinger Bands position.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            
        Returns:
            BB position (0-1) or None
        """
        if current_idx < self.bb_period:
            return None
        
        prices = bars['Close'].iloc[current_idx - self.bb_period:current_idx + 1]
        mean = prices.mean()
        std = prices.std()
        
        upper_band = mean + std * self.bb_std
        lower_band = mean - std * self.bb_std
        
        current_price = prices.iloc[-1]
        
        if upper_band == lower_band:
            return 0.5
        
        return (current_price - lower_band) / (upper_band - lower_band)
    
    def _has_reversion_confirmation(self, bars: Bars, current_idx: int, direction: int) -> bool:
        """Check for reversion confirmation in previous bars.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            direction: Direction to check (1 for oversold, -1 for overbought)
            
        Returns:
            True if reversion confirmed
        """
        if current_idx < self.min_reversion_bars:
            return False
        
        # Check if price has been moving in the extreme direction
        extreme_bars = 0
        for i in range(current_idx - self.min_reversion_bars, current_idx):
            if direction > 0:  # Oversold
                if self._get_rsi(bars, i) is not None and self._get_rsi(bars, i) < self.rsi_oversold:
                    extreme_bars += 1
            else:  # Overbought
                if self._get_rsi(bars, i) is not None and self._get_rsi(bars, i) > self.rsi_overbought:
                    extreme_bars += 1
        
        return extreme_bars >= self.min_reversion_bars - 1
    
    def _calculate_strength(self, bars: Bars, current_idx: int, direction: int) -> float:
        """Calculate signal strength.
        
        Args:
            bars: OHLCV data
            current_idx: Current bar index
            direction: Signal direction
            
        Returns:
            Signal strength (0-1)
        """
        # Base strength from RSI extreme
        rsi = self._get_rsi(bars, current_idx)
        if rsi is None:
            return 0.0
        
        if direction > 0:  # Oversold
            rsi_strength = (self.rsi_oversold - rsi) / self.rsi_oversold
        else:  # Overbought
            rsi_strength = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
        
        # Z-score strength
        zscore = self._get_zscore(bars, current_idx)
        if zscore is None:
            zscore_strength = 0.0
        else:
            zscore_strength = min(abs(zscore) / self.zscore_threshold, 1.0)
        
        # BB position strength
        bb_position = self._get_bb_position(bars, current_idx)
        if bb_position is None:
            bb_strength = 0.0
        else:
            if direction > 0:  # Oversold
                bb_strength = 1 - bb_position
            else:  # Overbought
                bb_strength = bb_position
        
        # Volume confirmation
        current_volume = bars['Volume'].iloc[current_idx]
        avg_volume = bars['Volume'].rolling(20).mean().iloc[current_idx]
        volume_strength = min(current_volume / avg_volume, 2.0) / 2.0
        
        # Combine strengths
        return (rsi_strength * 0.4 + zscore_strength * 0.3 + 
                bb_strength * 0.2 + volume_strength * 0.1)
    
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
        
        # Add technical indicators
        feature_dict['rsi'] = self._get_rsi(bars, current_idx)
        feature_dict['zscore'] = self._get_zscore(bars, current_idx)
        feature_dict['bb_position'] = self._get_bb_position(bars, current_idx)
        
        # Add computed features if available
        for col in features.columns:
            if current_idx < len(features):
                feature_dict[col] = features[col].iloc[current_idx]
        
        return feature_dict
    
    def validate_parameters(self) -> bool:
        """Validate setup parameters.
        
        Returns:
            True if parameters are valid
        """
        if self.rsi_period <= 0:
            return False
        
        if self.rsi_oversold >= self.rsi_overbought:
            return False
        
        if self.rsi_oversold <= 0 or self.rsi_overbought >= 100:
            return False
        
        if self.bb_period <= 0 or self.bb_std <= 0:
            return False
        
        if self.zscore_period <= 0 or self.zscore_threshold <= 0:
            return False
        
        if self.min_reversion_bars < 1:
            return False
        
        return True
    
    def get_required_features(self) -> List[str]:
        """Get list of required features for this setup.
        
        Returns:
            List of required feature names
        """
        return ['rsi_14', 'bb_width_20', 'zscore_20', 'volume_ratio_20']
    
    def get_description(self) -> str:
        """Get setup description.
        
        Returns:
            Setup description
        """
        return (
            "Mean Reversion setup identifies overbought/oversold conditions "
            "and looks for mean reversion opportunities. Uses RSI, Bollinger Bands, "
            "and Z-score indicators to identify extreme price levels."
        )

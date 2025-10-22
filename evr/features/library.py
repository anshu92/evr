"""Technical indicator library."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta


class FeatureLibrary:
    """Library of technical indicators and features."""
    
    @staticmethod
    def rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index).
        
        Args:
            data: OHLCV data
            period: RSI period
            
        Returns:
            RSI series
        """
        return ta.rsi(data['Close'], length=period)
    
    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR (Average True Range).
        
        Args:
            data: OHLCV data
            period: ATR period
            
        Returns:
            ATR series
        """
        return ta.atr(data['High'], data['Low'], data['Close'], length=period)
    
    @staticmethod
    def bb_width(data: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
        """Calculate Bollinger Bands width.
        
        Args:
            data: OHLCV data
            period: BB period
            std: Standard deviation multiplier
            
        Returns:
            BB width series
        """
        bb = ta.bbands(data['Close'], length=period, std=std)
        if bb is not None and 'BBW' in bb.columns:
            return bb['BBW']
        else:
            # Manual calculation if pandas_ta doesn't provide BBW
            bb_upper = ta.bbands(data['Close'], length=period, std=std)['BBU']
            bb_lower = ta.bbands(data['Close'], length=period, std=std)['BBL']
            return (bb_upper - bb_lower) / data['Close']
    
    @staticmethod
    def zscore(data: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
        """Calculate Z-score.
        
        Args:
            data: OHLCV data
            period: Z-score period
            column: Column to calculate Z-score for
            
        Returns:
            Z-score series
        """
        rolling_mean = data[column].rolling(window=period).mean()
        rolling_std = data[column].rolling(window=period).std()
        return (data[column] - rolling_mean) / rolling_std
    
    @staticmethod
    def adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index).
        
        Args:
            data: OHLCV data
            period: ADX period
            
        Returns:
            ADX series
        """
        adx_result = ta.adx(data['High'], data['Low'], data['Close'], length=period)
        if adx_result is not None and 'ADX' in adx_result.columns:
            return adx_result['ADX']
        else:
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def vwap(data: pd.DataFrame) -> pd.Series:
        """Calculate VWAP (Volume Weighted Average Price).
        
        Args:
            data: OHLCV data
            
        Returns:
            VWAP series
        """
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        return ta.vwap(typical_price, data['Volume'])
    
    @staticmethod
    def kama(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate KAMA (Kaufman's Adaptive Moving Average).
        
        Args:
            data: OHLCV data
            period: KAMA period
            
        Returns:
            KAMA series
        """
        return ta.kama(data['Close'], length=period)
    
    @staticmethod
    def macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Dictionary with MACD, signal, and histogram
        """
        macd_result = ta.macd(data['Close'], fast=fast, slow=slow, signal=signal)
        if macd_result is not None:
            return {
                'macd': macd_result['MACD'],
                'signal': macd_result['MACDs'],
                'histogram': macd_result['MACDh']
            }
        else:
            return {
                'macd': pd.Series(index=data.index, dtype=float),
                'signal': pd.Series(index=data.index, dtype=float),
                'histogram': pd.Series(index=data.index, dtype=float)
            }
    
    @staticmethod
    def sma(data: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
        """Calculate SMA (Simple Moving Average).
        
        Args:
            data: OHLCV data
            period: SMA period
            column: Column to calculate SMA for
            
        Returns:
            SMA series
        """
        return ta.sma(data[column], length=period)
    
    @staticmethod
    def ema(data: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
        """Calculate EMA (Exponential Moving Average).
        
        Args:
            data: OHLCV data
            period: EMA period
            column: Column to calculate EMA for
            
        Returns:
            EMA series
        """
        return ta.ema(data[column], length=period)
    
    @staticmethod
    def stoch(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator.
        
        Args:
            data: OHLCV data
            k_period: %K period
            d_period: %D period
            
        Returns:
            Dictionary with %K and %D
        """
        stoch_result = ta.stoch(data['High'], data['Low'], data['Close'], 
                               k=k_period, d=d_period)
        if stoch_result is not None:
            return {
                'stoch_k': stoch_result['STOCHk'],
                'stoch_d': stoch_result['STOCHd']
            }
        else:
            return {
                'stoch_k': pd.Series(index=data.index, dtype=float),
                'stoch_d': pd.Series(index=data.index, dtype=float)
            }
    
    @staticmethod
    def williams_r(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R.
        
        Args:
            data: OHLCV data
            period: Williams %R period
            
        Returns:
            Williams %R series
        """
        return ta.willr(data['High'], data['Low'], data['Close'], length=period)
    
    @staticmethod
    def cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate CCI (Commodity Channel Index).
        
        Args:
            data: OHLCV data
            period: CCI period
            
        Returns:
            CCI series
        """
        return ta.cci(data['High'], data['Low'], data['Close'], length=period)
    
    @staticmethod
    def roc(data: pd.DataFrame, period: int = 10, column: str = 'Close') -> pd.Series:
        """Calculate ROC (Rate of Change).
        
        Args:
            data: OHLCV data
            period: ROC period
            column: Column to calculate ROC for
            
        Returns:
            ROC series
        """
        return ta.roc(data[column], length=period)
    
    @staticmethod
    def momentum(data: pd.DataFrame, period: int = 10, column: str = 'Close') -> pd.Series:
        """Calculate Momentum.
        
        Args:
            data: OHLCV data
            period: Momentum period
            column: Column to calculate momentum for
            
        Returns:
            Momentum series
        """
        return data[column] / data[column].shift(period) - 1
    
    @staticmethod
    def volatility(data: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
        """Calculate rolling volatility.
        
        Args:
            data: OHLCV data
            period: Volatility period
            column: Column to calculate volatility for
            
        Returns:
            Volatility series
        """
        returns = data[column].pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)
    
    @staticmethod
    def price_position(data: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
        """Calculate price position within period range.
        
        Args:
            data: OHLCV data
            period: Period for range calculation
            column: Column to calculate position for
            
        Returns:
            Price position series (0-1)
        """
        rolling_min = data[column].rolling(window=period).min()
        rolling_max = data[column].rolling(window=period).max()
        return (data[column] - rolling_min) / (rolling_max - rolling_min)
    
    @staticmethod
    def volume_ratio(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate volume ratio.
        
        Args:
            data: OHLCV data
            period: Period for average volume calculation
            
        Returns:
            Volume ratio series
        """
        avg_volume = data['Volume'].rolling(window=period).mean()
        return data['Volume'] / avg_volume

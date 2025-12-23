import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

class DataEngine:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_tickers(self) -> List[str]:
        """Fetches the list of tickers to scan."""
        # Try to load from official_tickers.json if it exists
        ticker_file = os.path.join(self.cache_dir, "official_tickers.json")
        if os.path.exists(ticker_file):
            try:
                with open(ticker_file, "r") as f:
                    data = json.load(f)
                    return data.get("tickers", [])
            except Exception as e:
                print(f"Error reading ticker file: {e}")
        
        # Fallback to a default list if file doesn't exist or fails
        print("Using fallback ticker list (SPY, QQQ, IWM, AAPL, MSFT, etc.)")
        return ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]

    def fetch_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Fetches historical OHLCV data for a single ticker with caching."""
        cache_path = os.path.join(self.cache_dir, f"{ticker}_{period}.parquet")
        
        # Check cache first
        if os.path.exists(cache_path):
            try:
                # Check if cache is fresh (less than 24 hours old)
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
                if datetime.now() - mtime < timedelta(hours=24):
                    return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Cache read error for {ticker}: {e}")

        # Fetch from yfinance
        try:
            df = yf.download(ticker, period=period, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # Save to cache
                try:
                    df.to_parquet(cache_path)
                except Exception as e:
                    print(f"Cache write error for {ticker}: {e}")
                    
                return df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            
        return pd.DataFrame()

    def fetch_fundamentals(self, ticker: str) -> Dict[str, float]:
        """Fetches fundamental data for a ticker."""
        # Fundamentals are harder to cache effectively in a simple file without a DB, 
        # but we can wrap in try/except to be robust.
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return {
                "P/E": info.get("trailingPE", 0.0),
                "Forward P/E": info.get("forwardPE", 0.0),
                "Debt/Equity": info.get("debtToEquity", 0.0),
                "PEG": info.get("pegRatio", 0.0),
                "P/B": info.get("priceToBook", 0.0),
                "Market Cap": info.get("marketCap", 0.0),
                "Sector": info.get("sector", "Unknown")
            }
        except Exception as e:
            # print(f"Error fetching fundamentals for {ticker}: {e}") # Reduce noise
            return {}

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds technical indicators to the DataFrame."""
        if df.empty or len(df) < 30:
            return df
            
        df = df.copy()
        
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator (14)
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        # Avoid division by zero
        denom = high_14 - low_14
        df['%K'] = 100 * ((df['Close'] - low_14) / denom.replace(0, np.nan))

        # MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands (20, 2)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['20dSTD'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['MA20'] + (df['20dSTD'] * 2)
        df['Lower_Band'] = df['MA20'] - (df['20dSTD'] * 2)

        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        return df

    def prepare_model_input(self, df: pd.DataFrame, fundamentals: Dict[str, float]) -> Optional[str]:
        """
        Converts the latest row(s) into a text description for the T5 model.
        """
        if df.empty or len(df) < 30:
            return None
            
        # Get latest available data
        latest = df.iloc[-1]
        
        # Check for NaN in critical fields
        if pd.isna(latest['RSI']) or pd.isna(latest['MACD']):
            return None
        
        # Determine trend states
        macd_state = "Bullish" if latest['MACD'] > latest['Signal_Line'] else "Bearish"
        rsi_state = "Overbought" if latest['RSI'] > 70 else ("Oversold" if latest['RSI'] < 30 else "Neutral")
        price_vs_ma20 = "Above" if latest['Close'] > latest['MA20'] else "Below"
        
        # Fundamental summary
        fund_text = ", ".join([f"{k}: {v}" for k, v in fundamentals.items() if v and k != "Unknown"])

        prompt = (
            f"Analyze Market State: "
            f"RSI is {latest['RSI']:.1f} ({rsi_state}). "
            f"Stochastic %K is {latest['%K']:.1f}. "
            f"MACD is {macd_state} (Line: {latest['MACD']:.2f}, Signal: {latest['Signal_Line']:.2f}). "
            f"Price is {latest['Close']:.2f}, {price_vs_ma20} the 20-day MA. "
            f"Bollinger Bands are {latest['Lower_Band']:.2f} to {latest['Upper_Band']:.2f}. "
            f"Fundamentals: {fund_text}."
        )
        
        return prompt
    
    def get_training_data(self, limit: int = 50) -> Tuple[List[str], List[int]]:
        """
        Generates training examples (text, label) from historical data.
        Limits the number of tickers processed to avoid massive training times in this demo.
        """
        all_texts = []
        all_labels = []
        
        tickers = self.fetch_tickers()[:limit] # Train on a subset for speed/demo
        
        for ticker in tickers:
            df = self.fetch_data(ticker)
            if df.empty:
                continue
                
            fundamentals = self.fetch_fundamentals(ticker)
            df = self.add_technical_indicators(df)
            df.dropna(inplace=True)
            
            # Create targets: Next Close > Current Close (Simple Binary Classification)
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df.dropna(inplace=True) 
            
            # Downsample to avoid million-row datasets if not needed
            # Take last 200 days per stock
            df_sample = df.tail(200)
            
            for i in range(len(df_sample)):
                row = df_sample.iloc[i]
                
                macd_state = "Bullish" if row['MACD'] > row['Signal_Line'] else "Bearish"
                rsi_state = "Overbought" if row['RSI'] > 70 else ("Oversold" if row['RSI'] < 30 else "Neutral")
                price_vs_ma20 = "Above" if row['Close'] > row['MA20'] else "Below"
                fund_text = ", ".join([f"{k}: {v}" for k, v in fundamentals.items() if v])

                text = (
                    f"Analyze Market State for {ticker}: "
                    f"RSI is {row['RSI']:.1f} ({rsi_state}). "
                    f"Stochastic %K is {row['%K']:.1f}. "
                    f"MACD is {macd_state} (Line: {row['MACD']:.2f}, Signal: {row['Signal_Line']:.2f}). "
                    f"Price is {row['Close']:.2f}, {price_vs_ma20} the 20-day MA. "
                    f"Fundamentals: {fund_text}."
                )
                
                all_texts.append(text)
                all_labels.append(int(row['Target']))
                
        return all_texts, all_labels

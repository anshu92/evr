import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from market_timing.data_engine import DataEngine
from market_timing.model import MarketTimingClassifier

console = Console()

class MarketScanner:
    def __init__(self, 
                 model_path: str = "trained_parameters/market_timing_model.pt",
                 portfolio_file: str = "portfolio_state.json",
                 max_workers: int = 5):
        self.model_path = model_path
        self.portfolio_file = portfolio_file
        self.engine = DataEngine()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_workers = max_workers
        
        # Load Model
        self.model = MarketTimingClassifier().to(self.device)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            console.print(f"[green]Loaded model from {self.model_path}[/green]")
        else:
            console.print(f"[red]Model not found at {self.model_path}. Training required.[/red]")

    def _process_ticker(self, ticker: str) -> Dict[str, Any]:
        """Process a single ticker."""
        try:
            # Fetch Data
            df = self.engine.fetch_data(ticker)
            if df.empty:
                return None
                
            # Get Fundamentals
            fundamentals = self.engine.fetch_fundamentals(ticker)
            
            # Add Indicators
            df = self.engine.add_technical_indicators(df)
            
            # Prepare Input
            prompt = self.engine.prepare_model_input(df, fundamentals)
            if not prompt:
                return None
                
            # Return data needed for prediction (prediction happens in main thread for batching or simplicity)
            # Or predict here if model is thread-safe (it usually isn't without care)
            # Better approach: Return features, predict in batches in main thread.
            # However, T5 inference is fast enough for per-ticker if we lock or if we just return the prompt.
            
            return {
                "ticker": ticker,
                "prompt": prompt,
                "latest_price": float(df.iloc[-1]['Close']),
                "fundamentals": fundamentals
            }
                
        except Exception as e:
            return None

    def scan_market(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Scans the market for bullish setups using the T5 model with multiprocessing.
        """
        tickers = self.engine.fetch_tickers()
        if limit:
            tickers = tickers[:limit]
            
        results = []
        prompts_data = []
        
        console.print(f"Scanning {len(tickers)} tickers with {self.max_workers} workers...")
        
        # Step 1: Fetch and Prepare Data in Parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {executor.submit(self._process_ticker, ticker): ticker for ticker in tickers}
            
            for future in tqdm(as_completed(future_to_ticker), total=len(tickers), desc="Fetching Data"):
                res = future.result()
                if res:
                    prompts_data.append(res)
        
        # Step 2: Batch Prediction (Efficient on GPU/CPU)
        if not prompts_data:
            return []
            
        console.print(f"Running inference on {len(prompts_data)} valid tickers...")
        
        batch_size = 32
        for i in range(0, len(prompts_data), batch_size):
            batch = prompts_data[i:i+batch_size]
            prompts = [item['prompt'] for item in batch]
            
            try:
                # Predict batch
                preds, probs = self.model.predict(prompts, device=self.device)
                
                for j, item in enumerate(batch):
                    prob_bullish = float(probs[j][1])
                    
                    if prob_bullish > 0.6: # Confidence Threshold
                        results.append({
                            "ticker": item['ticker'],
                            "price": item['latest_price'],
                            "prob_bullish": prob_bullish,
                            "signal": "BULLISH",
                            "summary": item['prompt'],
                            "fundamentals": item['fundamentals']
                        })
            except Exception as e:
                console.print(f"[red]Error in batch inference: {e}[/red]")
                
        # Sort by confidence
        results.sort(key=lambda x: x['prob_bullish'], reverse=True)
        return results

class PortfolioManager:
    def __init__(self, portfolio_file: str = "portfolio_state.json", initial_capital: float = 10000.0):
        self.portfolio_file = portfolio_file
        self.initial_capital = initial_capital
        self.state = self._load_portfolio()

    def _load_portfolio(self) -> Dict[str, Any]:
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, "r") as f:
                return json.load(f)
        return {
            "total_capital": self.initial_capital,
            "available_capital": self.initial_capital,
            "allocated_capital": 0.0,
            "positions": [],
            "history": [],
            "last_updated": datetime.now().isoformat()
        }

    def save_portfolio(self):
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.portfolio_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def update_positions(self, current_prices: Dict[str, float]):
        """
        Updates open positions with current prices and checks exit conditions.
        Exit if:
        1. Stop loss hit (e.g., -5%)
        2. Profit target hit (e.g., +10%)
        3. Time exit (held > 7 days)
        """
        active_positions = []
        for pos in self.state["positions"]:
            ticker = pos["ticker"]
            if ticker not in current_prices:
                active_positions.append(pos)
                continue
                
            current_price = current_prices[ticker]
            entry_price = pos["entry_price"]
            shares = pos["shares"]
            
            # Calculate unrealized P&L
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Exit Logic
            exit_reason = None
            if pnl_pct < -0.05:
                exit_reason = "Stop Loss (-5%)"
            elif pnl_pct > 0.10:
                exit_reason = "Take Profit (+10%)"
            
            # Check time (simple 7 day hold)
            entry_date = datetime.fromisoformat(pos["entry_date"])
            days_held = (datetime.now() - entry_date).days
            if days_held >= 7 and not exit_reason:
                exit_reason = "Time Exit (7 days)"

            if exit_reason:
                # Close Position
                proceeds = shares * current_price
                pnl = proceeds - (shares * entry_price)
                
                self.state["available_capital"] += proceeds
                self.state["allocated_capital"] -= (shares * entry_price)
                self.state["history"].append({
                    "ticker": ticker,
                    "action": "SELL",
                    "price": current_price,
                    "shares": shares,
                    "date": datetime.now().isoformat(),
                    "reason": exit_reason,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct
                })
                console.print(f"[red]Selling {ticker}: {exit_reason} (PnL: {pnl_pct*100:.2f}%)[/red]")
            else:
                active_positions.append(pos)
        
        self.state["positions"] = active_positions
        self.save_portfolio()

    def allocate_new_positions(self, recommendations: List[Dict[str, Any]], max_positions: int = 5):
        """
        Allocates capital to new recommendations if slots are available.
        """
        current_positions = len(self.state["positions"])
        slots_available = max_positions - current_positions
        
        if slots_available <= 0:
            console.print("[yellow]Portfolio full. No new positions added.[/yellow]")
            return

        # Allocation per slot (simple equal weight)
        alloc_amount = self.state["available_capital"] / max(1, slots_available) 
        # Or strictly: total_capital / max_positions to maintain sizing
        target_size = self.state["total_capital"] / max_positions
        alloc_amount = min(alloc_amount, target_size)

        for rec in recommendations[:slots_available]:
            ticker = rec["ticker"]
            price = rec["price"]
            
            # Skip if already in portfolio
            if any(p["ticker"] == ticker for p in self.state["positions"]):
                continue
                
            shares = int(alloc_amount / price)
            if shares > 0:
                cost = shares * price
                self.state["available_capital"] -= cost
                self.state["allocated_capital"] += cost
                
                new_pos = {
                    "ticker": ticker,
                    "entry_price": price,
                    "shares": shares,
                    "entry_date": datetime.now().isoformat(),
                    "model_confidence": rec["prob_bullish"]
                }
                self.state["positions"].append(new_pos)
                
                self.state["history"].append({
                    "ticker": ticker,
                    "action": "BUY",
                    "price": price,
                    "shares": shares,
                    "date": datetime.now().isoformat(),
                    "reason": f"Model Confidence: {rec['prob_bullish']:.2f}"
                })
                
                console.print(f"[green]Buying {shares} shares of {ticker} at ${price:.2f}[/green]")
        
        self.save_portfolio()

    def generate_summary(self) -> str:
        """Generates a summary report of the portfolio."""
        summary = []
        summary.append(f"Total Capital: ${self.state['total_capital']:.2f}")
        summary.append(f"Available Cash: ${self.state['available_capital']:.2f}")
        summary.append(f"Open Positions: {len(self.state['positions'])}")
        
        if self.state['positions']:
            summary.append("\nPositions:")
            for pos in self.state['positions']:
                summary.append(f"- {pos['ticker']}: {pos['shares']} shares @ ${pos['entry_price']:.2f}")
                
        return "\n".join(summary)


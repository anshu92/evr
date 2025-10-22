#!/usr/bin/env python3
"""Minimal scan example demonstrating EVR usage."""

from evr.config import load_config
from evr.recommend import RecommendationScanner


def main():
    """Run a minimal scan example."""
    # Load configuration
    config = load_config()
    
    # Initialize scanner
    scanner = RecommendationScanner(config)
    
    # Run scan
    print("Scanning for trading recommendations...")
    trade_plans = scanner.scan(
        symbols=["AAPL", "MSFT", "GOOGL"],
        setups=["squeeze_breakout", "trend_pullback", "mean_reversion"],
        top_n=5,
    )
    
    # Display results
    if not trade_plans:
        print("No trading recommendations found")
        return
    
    print(f"\nFound {len(trade_plans)} trading recommendations:")
    print("-" * 80)
    
    for i, plan in enumerate(trade_plans, 1):
        direction = "LONG" if plan.signal.direction > 0 else "SHORT"
        print(f"{i}. {plan.signal.symbol} - {plan.signal.setup} - {direction}")
        print(f"   Entry: ${plan.entry_price:.2f}")
        print(f"   Stop Loss: ${plan.stop_loss:.2f}")
        print(f"   Take Profit: ${plan.take_profit:.2f}")
        print(f"   Position Size: ${plan.position_size:.2f}")
        print(f"   Probability: {plan.probability:.1%}")
        print(f"   Expected Return: {plan.expected_return:.2%}")
        print(f"   Signal Strength: {plan.signal.strength:.2f}")
        print()


if __name__ == "__main__":
    main()

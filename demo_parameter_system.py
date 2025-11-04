#!/usr/bin/env python3
"""
Demo Script: Historical Parameter Training System

This script demonstrates the complete workflow of training
and using historical parameters with the EVR scanner.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def print_section(title: str, content: str):
    """Print a formatted section."""
    console.print("\n")
    console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))
    console.print(content)
    console.print()


def demo_training():
    """Demonstrate the training process."""
    print_section(
        "Step 1: Training Parameters",
        """
This step fetches historical data and simulates trades to learn
probability distributions for different trading setups.

[yellow]In production, you would run:[/yellow]
    python run_parameter_training.py

[yellow]For a quick demo with fewer tickers:[/yellow]
    python run_parameter_training.py --tickers AAPL MSFT GOOGL --lookback 6

[green]This will:[/green]
  1. Fetch 6 months of data for AAPL, MSFT, GOOGL
  2. Generate signals (RSI, MACD, BB, etc.)
  3. Simulate trades (entry → exit)
  4. Calculate statistics
  5. Save parameters to trained_parameters/
        """
    )


def demo_analysis():
    """Demonstrate parameter analysis."""
    print_section(
        "Step 2: Analyzing Results",
        """
After training, review the learned parameters:

[yellow]Command:[/yellow]
    python run_parameter_training.py --mode analyze

[green]You'll see:[/green]
  • Overall statistics (win rate, expectancy)
  • Setup-specific parameters
  • Number of trades per setup
  • Profit factors and R-multiples

[cyan]Example Output:[/cyan]
    RSI_Oversold_Long: 58% win rate, +2.6% expectancy
    MACD_Cross_Long: 52% win rate, +1.2% expectancy
    BB_Bounce_Long: 57% win rate, +2.1% expectancy
        """
    )


def demo_comparison():
    """Demonstrate parameter comparison."""
    print_section(
        "Step 3: Comparing with Defaults",
        """
See how trained parameters compare to default assumptions:

[yellow]Command:[/yellow]
    python run_parameter_training.py --mode compare

[green]Comparison:[/green]
    Parameter    Default    Trained    Difference
    ─────────────────────────────────────────────
    P(Win)       50.0%      54.2%      +4.2%
    Avg Win       5.0%       6.8%      +1.8%
    Avg Loss     -3.0%      -4.2%      -1.2%
    Expectancy   +1.0%      +1.76%     +0.76%

[cyan]Interpretation:[/cyan]
  • Higher win rate than assumed
  • Larger wins but also larger losses
  • Better overall expectancy
        """
    )


def demo_integration():
    """Demonstrate scanner integration."""
    print_section(
        "Step 4: Using with Scanner",
        """
The scanner automatically loads trained parameters!

[yellow]Simply run:[/yellow]
    python official_scanner.py

[green]Scanner logs:[/green]
    ✓ Loaded trained parameters from historical backtesting
    Available trained setups: RSI_Oversold_Long, MACD_Cross_Long, ...

[cyan]How it works:[/cyan]
  1. Scanner checks for trained_parameters/scanner_parameters.json
  2. If found, loads and wraps probability model
  3. Blends trained params with real trading data
  4. As real trades accumulate, shifts from trained → real
        """
    )


def demo_adaptive_learning():
    """Demonstrate adaptive learning."""
    print_section(
        "Step 5: Adaptive Learning",
        """
The scanner adapts as it accumulates real trading data:

[yellow]Day 1 (0 real trades):[/yellow]
    P(Win) = 58.4% (100% from training)
    Weight: trained=100%, real=0%

[yellow]Day 30 (15 real trades):[/yellow]
    Trained: P(Win) = 58.4%
    Real:    P(Win) = 62.0%
    Blended: P(Win) = 60.2%
    Weight: trained=50%, real=50%

[yellow]Day 90 (30+ real trades):[/yellow]
    P(Win) = 62.0% (100% from real trades)
    Weight: trained=0%, real=100%

[green]Benefits:[/green]
  • Start with informed priors
  • Gradually adapt to reality
  • Smooth transition
  • No sudden jumps
        """
    )


def demo_file_structure():
    """Show file structure."""
    print_section(
        "File Structure",
        """
After training, you'll have:

[cyan]evr/[/cyan]
  ├── [yellow]trained_parameters/[/yellow]
  │   ├── scanner_parameters.json      [green]← Scanner loads this[/green]
  │   ├── trained_statistics.json      [green]← Full stats[/green]
  │   ├── trade_results.csv            [green]← All simulated trades[/green]
  │   └── trained_parameters.pkl       [green]← Python pickle[/green]
  │
  ├── [yellow]Core Training Modules:[/yellow]
  │   ├── historical_parameter_trainer.py
  │   ├── parameter_integration.py
  │   └── run_parameter_training.py
  │
  ├── [yellow]Scanner:[/yellow]
  │   └── official_scanner.py          [green]← Auto-loads params[/green]
  │
  └── [yellow]Documentation:[/yellow]
      ├── PARAMETER_TRAINING_README.md
      └── QUICKSTART_PARAMETER_TRAINING.md
        """
    )


def demo_code_example():
    """Show code example."""
    print_section(
        "Code Example: Manual Integration",
        """
If you want explicit control over parameter loading:

[yellow]Python code:[/yellow]

    from official_scanner import OfficialTickerScanner
    from parameter_integration import integrate_trained_parameters
    
    # Create scanner
    scanner = OfficialTickerScanner(
        initial_capital=10000,
        use_ml_classifier=True
    )
    
    # Integrate trained parameters
    integrate_trained_parameters(
        scanner, 
        "trained_parameters/scanner_parameters.json"
    )
    
    # Run scanner
    plans = scanner.scan(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        max_positions=5
    )
    
    # Scanner now uses trained probabilities!

[green]Or just run the scanner - it auto-loads parameters![/green]
        """
    )


def demo_customization():
    """Show customization options."""
    print_section(
        "Customization Options",
        """
[yellow]1. Custom Tickers:[/yellow]
    python run_parameter_training.py --tickers AAPL TSLA NVDA AMD

[yellow]2. Longer History:[/yellow]
    python run_parameter_training.py --lookback 24  # 2 years

[yellow]3. Custom Output:[/yellow]
    python run_parameter_training.py --output my_params

[yellow]4. Analyze Only:[/yellow]
    python run_parameter_training.py --mode analyze

[yellow]5. Compare Only:[/yellow]
    python run_parameter_training.py --mode compare

[yellow]6. Full Pipeline:[/yellow]
    python run_parameter_training.py --mode all  # Default
        """
    )


def demo_validation():
    """Show validation checklist."""
    print_section(
        "Validation Checklist",
        """
[yellow]Check these after training:[/yellow]

[green]✓ Sample Size[/green]
  • Total trades > 500
  • Trades per setup > 50
  • If less: add more tickers or increase lookback

[green]✓ Statistical Significance[/green]
  • P(Win) should be 40-70% (outside this is suspicious)
  • Expectancy can be negative for some setups (that's OK!)
  • Profit Factor > 1.5 is excellent

[green]✓ Data Quality[/green]
  • No extreme outliers (>50% single trade return)
  • Reasonable stop distances (1-5% typical)
  • Target/stop ratio makes sense (1:2 to 1:4)

[green]✓ Setup Diversity[/green]
  • Multiple setups should show positive expectancy
  • If all negative: market conditions may be unfavorable
  • Different setups should have different characteristics
        """
    )


def demo_best_practices():
    """Show best practices."""
    print_section(
        "Best Practices",
        """
[yellow]1. Regular Retraining[/yellow]
   • Monthly or quarterly
   • Markets evolve, parameters should too
   • Keep old versions for comparison

[yellow]2. Diverse Training Set[/yellow]
   • 20-50 tickers across sectors
   • Not just tech stocks!
   • Include different volatility profiles

[yellow]3. Sufficient History[/yellow]
   • Minimum 6 months
   • Recommended 12+ months
   • Include different market regimes

[yellow]4. Monitor Performance[/yellow]
   • Track real trades vs predictions
   • Compare actual vs expected returns
   • Retrain if drift detected

[yellow]5. Walk-Forward Testing[/yellow]
   • Train on period 1, test on period 2
   • Verify parameters generalize
   • Don't overfit to specific period

[yellow]6. Risk Management[/yellow]
   • Parameters improve priors, not guarantees
   • Use proper position sizing
   • Respect stop losses
   • Diversify across setups
        """
    )


def demo_troubleshooting():
    """Show troubleshooting guide."""
    print_section(
        "Troubleshooting",
        """
[red]Problem:[/red] Training takes too long (>30 min)
[green]Solution:[/green] Reduce tickers or lookback period

[red]Problem:[/red] "Insufficient data for ticker"
[green]Solution:[/green] Some tickers are newly listed, remove or increase lookback

[red]Problem:[/red] All setups show negative expectancy
[green]Solution:[/green] 
  • Check date range (bear market?)
  • Verify signal logic
  • Try different tickers

[red]Problem:[/red] Scanner doesn't load parameters
[green]Solution:[/green] 
  • Check trained_parameters/ exists
  • Verify scanner_parameters.json is present
  • Check file permissions

[red]Problem:[/red] Very low sample sizes (<100 total trades)
[green]Solution:[/green]
  • Add more tickers
  • Increase lookback period
  • Check if signals are too restrictive
        """
    )


def main():
    """Main demo."""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Historical Parameter Training System - Demo[/bold cyan]\n"
        "Complete workflow demonstration",
        border_style="cyan"
    ))
    
    # Show all sections
    demo_training()
    demo_analysis()
    demo_comparison()
    demo_integration()
    demo_adaptive_learning()
    demo_file_structure()
    demo_code_example()
    demo_customization()
    demo_validation()
    demo_best_practices()
    demo_troubleshooting()
    
    # Final message
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]Ready to Start![/bold green]\n\n"
        "[yellow]Quick Start:[/yellow]\n"
        "1. Install: pip install -r requirements_training.txt\n"
        "2. Train:   python run_parameter_training.py\n"
        "3. Scan:    python official_scanner.py\n\n"
        "[cyan]For detailed docs:[/cyan]\n"
        "• QUICKSTART_PARAMETER_TRAINING.md\n"
        "• PARAMETER_TRAINING_README.md",
        border_style="green"
    ))


if __name__ == "__main__":
    main()



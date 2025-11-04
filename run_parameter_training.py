#!/usr/bin/env python3
"""
Complete Workflow for Training and Using Historical Parameters

This script:
1. Trains parameters from historical data
2. Shows how to integrate them into the scanner
3. Provides analysis and comparison tools
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from historical_parameter_trainer import HistoricalParameterTrainer
from parameter_integration import TrainedParameterLoader

console = Console()


def train_parameters(tickers: list, lookback_months: int = 12, output_dir: str = "trained_parameters"):
    """Train parameters from historical data.
    
    Args:
        tickers: List of ticker symbols
        lookback_months: Number of months of history
        output_dir: Output directory for trained parameters
    """
    console.print(Panel.fit(
        "[bold cyan]Step 1: Training Parameters from Historical Data[/bold cyan]",
        border_style="cyan"
    ))
    
    trainer = HistoricalParameterTrainer(
        tickers=tickers,
        lookback_months=lookback_months,
        output_dir=output_dir
    )
    
    trainer.run()
    
    return trainer


def analyze_trained_parameters(parameters_path: str = "trained_parameters/scanner_parameters.json"):
    """Analyze and display trained parameters.
    
    Args:
        parameters_path: Path to trained parameters file
    """
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Step 2: Analyzing Trained Parameters[/bold cyan]",
        border_style="cyan"
    ))
    
    loader = TrainedParameterLoader(parameters_path)
    
    if not loader.is_loaded:
        console.print("[red]Error: Could not load parameters![/red]")
        return
    
    # Display metadata
    metadata = loader.parameters.get('metadata', {})
    
    info_table = Table(title="Training Information", box=box.ROUNDED)
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="green")
    
    info_table.add_row("Training Date", metadata.get('training_date', 'N/A'))
    info_table.add_row("Lookback Period", f"{metadata.get('lookback_months', 0)} months")
    info_table.add_row("Number of Tickers", str(metadata.get('num_tickers', 0)))
    info_table.add_row("Total Trades", str(metadata.get('num_trades', 0)))
    
    console.print(info_table)
    console.print()
    
    # Display global prior
    p_win, avg_win, avg_loss = loader.get_global_prior()
    
    prior_table = Table(title="Global Prior Parameters", box=box.ROUNDED)
    prior_table.add_column("Parameter", style="cyan")
    prior_table.add_column("Value", style="green")
    
    prior_table.add_row("P(Win)", f"{p_win:.2%}")
    prior_table.add_row("Avg Win", f"{avg_win:.2%}")
    prior_table.add_row("Avg Loss", f"{avg_loss:.2%}")
    
    alpha, beta = loader.get_beta_prior()
    prior_table.add_row("Beta Alpha", f"{alpha:.2f}")
    prior_table.add_row("Beta Beta", f"{beta:.2f}")
    
    console.print(prior_table)
    console.print()
    
    # Display setup-specific parameters
    setups = loader.get_all_setups()
    
    setup_table = Table(title="Setup-Specific Parameters", box=box.ROUNDED, show_lines=True)
    setup_table.add_column("Setup", style="cyan")
    setup_table.add_column("Trades", justify="right")
    setup_table.add_column("P(Win)", justify="right")
    setup_table.add_column("Avg Win", justify="right")
    setup_table.add_column("Avg Loss", justify="right")
    setup_table.add_column("Expectancy", justify="right")
    setup_table.add_column("Profit Factor", justify="right")
    
    for setup in setups:
        params = loader.get_setup_parameters(setup)
        if params:
            style = "green" if params['expectancy'] > 0 else "red"
            setup_table.add_row(
                setup,
                str(params['total_trades']),
                f"{params['p_win']:.1%}",
                f"{params['avg_win']:.1%}",
                f"{params['avg_loss']:.1%}",
                f"[{style}]{params['expectancy']:.2%}[/{style}]",
                f"{params['profit_factor']:.2f}"
            )
    
    console.print(setup_table)


def compare_parameters(trained_path: str = "trained_parameters/scanner_parameters.json"):
    """Compare trained parameters with default parameters.
    
    Args:
        trained_path: Path to trained parameters
    """
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Step 3: Comparing with Default Parameters[/bold cyan]",
        border_style="cyan"
    ))
    
    loader = TrainedParameterLoader(trained_path)
    
    if not loader.is_loaded:
        console.print("[red]Error: Could not load parameters![/red]")
        return
    
    # Get trained parameters
    trained_p, trained_win, trained_loss = loader.get_global_prior()
    
    # Default parameters (from RollingBayes in official_scanner.py)
    default_p = 0.50
    default_win = 0.05
    default_loss = -0.03
    
    comp_table = Table(title="Default vs Trained Parameters", box=box.ROUNDED)
    comp_table.add_column("Parameter", style="cyan")
    comp_table.add_column("Default", style="yellow", justify="right")
    comp_table.add_column("Trained", style="green", justify="right")
    comp_table.add_column("Difference", style="magenta", justify="right")
    
    # P(Win)
    p_diff = trained_p - default_p
    comp_table.add_row(
        "P(Win)",
        f"{default_p:.1%}",
        f"{trained_p:.1%}",
        f"{p_diff:+.1%}"
    )
    
    # Avg Win
    win_diff = trained_win - default_win
    comp_table.add_row(
        "Avg Win",
        f"{default_win:.1%}",
        f"{trained_win:.1%}",
        f"{win_diff:+.1%}"
    )
    
    # Avg Loss
    loss_diff = trained_loss - default_loss
    comp_table.add_row(
        "Avg Loss",
        f"{default_loss:.1%}",
        f"{trained_loss:.1%}",
        f"{loss_diff:+.1%}"
    )
    
    # Expectancy
    default_exp = default_p * default_win + (1 - default_p) * default_loss
    trained_exp = trained_p * trained_win + (1 - trained_p) * trained_loss
    exp_diff = trained_exp - default_exp
    
    comp_table.add_row(
        "Expectancy",
        f"{default_exp:.2%}",
        f"{trained_exp:.2%}",
        f"{exp_diff:+.2%}"
    )
    
    console.print(comp_table)
    
    # Impact analysis
    console.print("\n[bold]Impact Analysis:[/bold]")
    
    if trained_exp > default_exp:
        console.print(f"✓ [green]Trained parameters show {exp_diff:.2%} better expectancy![/green]")
    else:
        console.print(f"⚠ [yellow]Trained parameters show {exp_diff:.2%} lower expectancy[/yellow]")
    
    if trained_p > default_p:
        console.print(f"✓ [green]Win rate is {p_diff:.1%} higher than default[/green]")
    else:
        console.print(f"⚠ [yellow]Win rate is {abs(p_diff):.1%} lower than default[/yellow]")


def generate_integration_code():
    """Generate code snippet for integrating parameters into scanner."""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Step 4: Integration with Scanner[/bold cyan]",
        border_style="cyan"
    ))
    
    code = '''
# In your scanner code (official_scanner.py), add at the top:
from parameter_integration import integrate_trained_parameters

# After creating the scanner instance, add:
scanner = EVRScanner(...)

# Integrate trained parameters
integrate_trained_parameters(scanner, "trained_parameters/scanner_parameters.json")

# Now the scanner will use trained parameters as priors!
# The scanner will blend trained parameters with real-time data
# as it accumulates actual trading results.
'''
    
    console.print(Panel(code, title="Integration Code", border_style="green"))
    
    console.print("\n[bold]Alternative: Automatic Loading[/bold]")
    console.print("You can modify the scanner to automatically load parameters on startup.")
    console.print("See the updated scanner code in official_scanner.py")


def create_scanner_patch():
    """Create a patch file for the scanner to auto-load parameters."""
    patch_content = '''
# Add this to the EVRScanner.__init__ method:

# Optional: Load trained parameters
if Path("trained_parameters/scanner_parameters.json").exists():
    from parameter_integration import integrate_trained_parameters
    integrate_trained_parameters(self, "trained_parameters/scanner_parameters.json")
    self.logger.info("Loaded trained parameters from historical data")
'''
    
    patch_path = Path("trained_parameters/scanner_integration_patch.txt")
    patch_path.parent.mkdir(exist_ok=True)
    
    with open(patch_path, 'w') as f:
        f.write(patch_content)
    
    console.print(f"\n[green]Created integration patch at {patch_path}[/green]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train and integrate historical parameters for EVR Scanner"
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'analyze', 'compare', 'all'],
        default='all',
        help="Mode: train, analyze, compare, or all"
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help="List of tickers to train on (default: predefined list)"
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=12,
        help="Lookback period in months (default: 12)"
    )
    
    parser.add_argument(
        '--output',
        default="trained_parameters",
        help="Output directory (default: trained_parameters)"
    )
    
    args = parser.parse_args()
    
    # Default tickers if not provided
    if not args.tickers:
        args.tickers = [
            # Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'INTC',
            # Finance
            'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK',
            # Consumer
            'WMT', 'HD', 'DIS', 'NKE', 'SBUX', 'MCD',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB',
            # Industrial
            'BA', 'CAT', 'GE', 'HON', 'MMM',
        ]
    
    # Welcome message
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]EVR Historical Parameter Training System[/bold cyan]\n"
        "Train probability models from historical backtesting data",
        border_style="cyan"
    ))
    
    # Execute based on mode
    if args.mode in ['train', 'all']:
        train_parameters(args.tickers, args.lookback, args.output)
    
    if args.mode in ['analyze', 'all']:
        analyze_trained_parameters(f"{args.output}/scanner_parameters.json")
    
    if args.mode in ['compare', 'all']:
        compare_parameters(f"{args.output}/scanner_parameters.json")
    
    if args.mode == 'all':
        generate_integration_code()
        create_scanner_patch()
    
    # Summary
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]✓ Complete![/bold green]\n\n"
        "Next steps:\n"
        "1. Review the trained parameters in trained_parameters/\n"
        "2. Integrate with scanner using parameter_integration.py\n"
        "3. Monitor performance as scanner accumulates real trades",
        border_style="green"
    ))


if __name__ == "__main__":
    main()



"""CLI interface for EVR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from evr.config import load_config
from evr.backtest import BacktestEngine
from evr.recommend import RecommendationScanner
from evr.reporting import ReportGenerator

app = typer.Typer(
    name="evr",
    help="EVR - Empirical Volatility Regime Trading Framework",
    add_completion=False,
)

console = Console()


@app.command()
def data(
    download: bool = typer.Option(False, "--download", "-d", help="Download data"),
    tickers: Optional[str] = typer.Option(None, "--tickers", "-t", help="Comma-separated list of tickers"),
    start: Optional[str] = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD)"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
):
    """Manage data downloads and caching."""
    if not download:
        console.print("Use --download flag to download data")
        return
    
    if not tickers:
        console.print("Please specify tickers with --tickers")
        return
    
    # Load configuration
    config = load_config(config_path)
    
    # Parse tickers
    ticker_list = [t.strip() for t in tickers.split(",")]
    
    # Set default dates if not provided
    if not start:
        start = "2020-01-01"
    if not end:
        end = "2025-01-01"
    
    console.print(f"Downloading data for {len(ticker_list)} tickers from {start} to {end}")
    
    # TODO: Implement data download
    console.print("Data download functionality will be implemented")


@app.command()
def scan(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    asof: Optional[str] = typer.Option(None, "--asof", "-a", help="As-of date (YYYY-MM-DD)"),
    symbols: Optional[str] = typer.Option(None, "--symbols", "-s", help="Comma-separated list of symbols"),
    setups: Optional[str] = typer.Option(None, "--setups", help="Comma-separated list of setups"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of top recommendations"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Scan for trading recommendations."""
    # Load configuration
    config = load_config(config_path)
    
    # Parse symbols
    symbol_list = None
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    
    # Parse setups
    setup_list = None
    if setups:
        setup_list = [s.strip() for s in setups.split(",")]
    
    console.print("Scanning for trading recommendations...")
    
    # Initialize scanner
    scanner = RecommendationScanner(config)
    
    # Run scan
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning...", total=None)
        
        try:
            trade_plans = scanner.scan(
                symbols=symbol_list,
                setups=setup_list,
                asof_date=asof,
                top_n=top_n,
            )
            progress.update(task, description="Scan completed")
        except Exception as e:
            console.print(f"[red]Error during scan: {e}[/red]")
            return
    
    # Display results
    if not trade_plans:
        console.print("[yellow]No trading recommendations found[/yellow]")
        return
    
    # Create table
    table = Table(title="Trading Recommendations")
    table.add_column("Symbol", style="cyan")
    table.add_column("Setup", style="magenta")
    table.add_column("Direction", style="green")
    table.add_column("Entry Price", style="yellow")
    table.add_column("Stop Loss", style="red")
    table.add_column("Take Profit", style="green")
    table.add_column("Position Size", style="blue")
    table.add_column("Probability", style="cyan")
    table.add_column("Expected Return", style="green")
    
    for plan in trade_plans:
        direction = "LONG" if plan.signal.direction > 0 else "SHORT"
        table.add_row(
            plan.signal.symbol,
            plan.signal.setup,
            direction,
            f"${plan.entry_price:.2f}",
            f"${plan.stop_loss:.2f}",
            f"${plan.take_profit:.2f}",
            f"${plan.position_size:.2f}",
            f"{plan.probability:.1%}",
            f"{plan.expected_return:.2%}",
        )
    
    console.print(table)
    
    # Save to file if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        json_data = []
        for plan in trade_plans:
            json_data.append({
                'symbol': plan.signal.symbol,
                'setup': plan.signal.setup,
                'direction': plan.signal.direction,
                'entry_price': plan.entry_price,
                'stop_loss': plan.stop_loss,
                'take_profit': plan.take_profit,
                'position_size': plan.position_size,
                'probability': plan.probability,
                'expected_return': plan.expected_return,
                'signal_strength': plan.signal.strength,
                'features': plan.signal.features,
                'metadata': plan.signal.metadata,
            })
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        console.print(f"[green]Results saved to {output_path}[/green]")


@app.command()
def backtest(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    start: Optional[str] = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD)"),
    symbols: Optional[str] = typer.Option(None, "--symbols", help="Comma-separated list of symbols"),
    setups: Optional[str] = typer.Option(None, "--setups", help="Comma-separated list of setups"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Run backtest."""
    # Load configuration
    config = load_config(config_path)
    
    # Parse symbols
    symbol_list = None
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    
    # Parse setups
    setup_list = None
    if setups:
        setup_list = [s.strip() for s in setups.split(",")]
    
    # Set default dates if not provided
    if not start:
        start = "2020-01-01"
    if not end:
        end = "2025-01-01"
    
    console.print(f"Running backtest from {start} to {end}")
    
    # Initialize backtest engine
    engine = BacktestEngine(config)
    
    # Run backtest
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running backtest...", total=None)
        
        try:
            results = engine.run_backtest(
                symbols=symbol_list or ["AAPL", "MSFT", "GOOGL"],
                start_date=start,
                end_date=end,
                setups=setup_list,
            )
            progress.update(task, description="Backtest completed")
        except Exception as e:
            console.print(f"[red]Error during backtest: {e}[/red]")
            return
    
    # Display results
    metrics = results.get('metrics', {})
    
    # Create metrics table
    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Return", f"{metrics.total_return:.2%}")
    table.add_row("CAGR", f"{metrics.cagr:.2%}")
    table.add_row("Max Drawdown", f"{metrics.max_drawdown:.2%}")
    table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    table.add_row("Win Rate", f"{metrics.win_rate:.1%}")
    table.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
    table.add_row("Total Trades", str(metrics.total_trades))
    table.add_row("Winning Trades", str(metrics.winning_trades))
    table.add_row("Losing Trades", str(metrics.losing_trades))
    
    console.print(table)
    
    # Generate report if output directory specified
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize report generator
        report_generator = ReportGenerator(config)
        
        # Generate report
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating report...", total=None)
            
            try:
                report_path = report_generator.generate_report(
                    run_id=results['run_id'],
                    results=results,
                )
                progress.update(task, description="Report generated")
            except Exception as e:
                console.print(f"[red]Error generating report: {e}[/red]")
                return
        
        console.print(f"[green]Report generated at {report_path}[/green]")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Generate report for a specific run."""
    # Load configuration
    config = load_config(config_path)
    
    # Set output directory
    if output_dir:
        config.report.output_dir = output_dir
    
    console.print(f"Generating report for run {run_id}")
    
    # TODO: Load results from run_id
    # For now, create a dummy results structure
    results = {
        'run_id': run_id,
        'config': config.to_dict(),
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'start_date': '2020-01-01',
        'end_date': '2025-01-01',
        'setups': ['squeeze_breakout', 'trend_pullback', 'mean_reversion'],
        'trade_results': [],
        'equity_curve': [],
        'timestamps': [],
        'metrics': {},
    }
    
    # Initialize report generator
    report_generator = ReportGenerator(config)
    
    # Generate report
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating report...", total=None)
        
        try:
            report_path = report_generator.generate_report(
                run_id=run_id,
                results=results,
            )
            progress.update(task, description="Report generated")
        except Exception as e:
            console.print(f"[red]Error generating report: {e}[/red]")
            return
    
    console.print(f"[green]Report generated at {report_path}[/green]")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    validate: bool = typer.Option(False, "--validate", "-v", help="Validate configuration"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
):
    """Manage configuration."""
    if show:
        # Load and display configuration
        config = load_config(config_path)
        
        # Create configuration panel
        config_text = f"""
Name: {config.name}
Description: {config.description}
Version: {config.version}

Data Configuration:
  Primary Source: {config.data.primary_source}
  Cache Directory: {config.data.cache_dir}
  Timeframe: {config.data.timeframe}

Risk Configuration:
  Initial Capital: ${config.risk.initial_capital:,.2f}
  Max Position Size: {config.risk.max_position_size:.1%}
  Max Risk Per Trade: {config.risk.max_risk_per_trade:.1%}
  Kelly Fraction: {config.risk.kelly_fraction:.1%}

Probability Configuration:
  Window Size: {config.prob.window_size}
  Min Samples: {config.prob.min_samples}
  Alpha: {config.prob.alpha}
        """
        
        panel = Panel(config_text, title="EVR Configuration", border_style="blue")
        console.print(panel)
    
    elif validate:
        # Validate configuration
        try:
            config = load_config(config_path)
            config.validate()
            console.print("[green]Configuration is valid[/green]")
        except Exception as e:
            console.print(f"[red]Configuration validation failed: {e}[/red]")
    
    else:
        console.print("Use --show to display configuration or --validate to validate it")


@app.command()
def version():
    """Show version information."""
    from evr import __version__
    
    version_text = f"""
EVR - Empirical Volatility Regime Trading Framework
Version: {__version__}

A quantitative trading framework for empirical volatility regime analysis.
    """
    
    panel = Panel(version_text, title="Version Information", border_style="green")
    console.print(panel)


if __name__ == "__main__":
    app()

#!/usr/bin/env python3
"""
CLI interface for the official EVR ticker scanner.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from official_scanner import OfficialTickerScanner


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(
        description="EVR Official Ticker Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_official_scanner.py                           # Run with default settings
  python cli_official_scanner.py --max-tickers 200         # Scan 200 tickers
  python cli_official_scanner.py --top 30                 # Show top 30 signals
  python cli_official_scanner.py --log-level DEBUG        # Enable debug logging
  python cli_official_scanner.py --no-cache               # Force fresh ticker fetch
        """
    )
    
    parser.add_argument(
        "--max-tickers", "-m",
        type=int,
        default=100,
        help="Maximum number of tickers to scan (default: 100)"
    )
    
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=20,
        help="Number of top signals to display (default: 20)"
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="signals",
        help="Prefix for output files (default: signals)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh ticker fetch (ignore cache)"
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Display configuration
    config_text = f"""
Configuration:
  Max Tickers: {args.max_tickers}
  Top Signals: {args.top}
  Output Prefix: {args.output_prefix}
  Log Level: {args.log_level}
  Use Cache: {not args.no_cache}
    """
    
    panel = Panel(config_text, title="EVR Official Scanner Configuration", border_style="blue")
    console.print(panel)
    
    try:
        # Initialize scanner with logging
        scanner = OfficialTickerScanner(log_level=args.log_level)
        
        # Get ticker list
        console.print("\n[blue]Getting official ticker lists...[/blue]")
        tickers = scanner.get_comprehensive_tickers(use_cache=not args.no_cache)
        console.print(f"[green]Found {len(tickers)} official tickers[/green]")
        
        # Scan for signals
        console.print(f"\n[blue]Scanning {min(args.max_tickers, len(tickers))} tickers...[/blue]")
        signals = scanner.scan_tickers(tickers, max_tickers=args.max_tickers)
        
        # Display results
        console.print(f"\n[blue]Displaying top {args.top} results...[/blue]")
        scanner.display_results(signals, top_n=args.top)
        
        # Save results
        console.print("\n[blue]Saving results...[/blue]")
        scanner.save_results(signals, args.output_prefix)
        
        console.print("\n[green]✅ Scan completed successfully![/green]")
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Scan interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]❌ Error during scan: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())

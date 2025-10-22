"""Report generator for EVR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jinja2 import Environment, FileSystemLoader

from ...types import Metrics, TradeResult


class ReportGenerator:
    """Generate reports for backtest results."""
    
    def __init__(self, config):
        """Initialize report generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.template_dir = Path(__file__).parent / "templates"
        self.output_dir = Path(config.report.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True
        )
        
        # Set up matplotlib style
        plt.style.use(config.report.chart_style)
        sns.set_palette("husl")
    
    def generate_report(
        self,
        run_id: str,
        results: Dict[str, Any],
        output_format: Optional[str] = None,
    ) -> str:
        """Generate comprehensive report.
        
        Args:
            run_id: Run ID
            results: Backtest results
            output_format: Output format (html, markdown, json)
            
        Returns:
            Path to generated report
        """
        if output_format is None:
            output_format = self.config.report.format
        
        # Create run directory
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate charts if requested
        if self.config.report.include_charts:
            self._generate_charts(run_dir, results)
        
        # Generate report based on format
        if output_format == "html":
            report_path = self._generate_html_report(run_dir, results)
        elif output_format == "markdown":
            report_path = self._generate_markdown_report(run_dir, results)
        elif output_format == "json":
            report_path = self._generate_json_report(run_dir, results)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return str(report_path)
    
    def _generate_charts(self, run_dir: Path, results: Dict[str, Any]) -> None:
        """Generate charts for the report.
        
        Args:
            run_dir: Run directory
            results: Backtest results
        """
        charts_dir = run_dir / "charts"
        charts_dir.mkdir(exist_ok=True)
        
        # Equity curve chart
        if self.config.report.include_equity_curve:
            self._create_equity_curve_chart(charts_dir, results)
        
        # Drawdown chart
        if self.config.report.include_drawdown:
            self._create_drawdown_chart(charts_dir, results)
        
        # Returns histogram
        if self.config.report.include_returns_histogram:
            self._create_returns_histogram(charts_dir, results)
        
        # Trade analysis
        if self.config.report.include_trade_analysis:
            self._create_trade_analysis_charts(charts_dir, results)
        
        # Setup analysis
        if self.config.report.include_setup_analysis:
            self._create_setup_analysis_charts(charts_dir, results)
    
    def _create_equity_curve_chart(self, charts_dir: Path, results: Dict[str, Any]) -> None:
        """Create equity curve chart.
        
        Args:
            charts_dir: Charts directory
            results: Backtest results
        """
        equity_curve = results.get('equity_curve', [])
        timestamps = results.get('timestamps', [])
        
        if not equity_curve or not timestamps:
            return
        
        fig, ax = plt.subplots(figsize=self.config.report.chart_size)
        
        # Convert to pandas for easier plotting
        df = pd.DataFrame({
            'equity': equity_curve,
            'timestamp': timestamps
        })
        df.set_index('timestamp', inplace=True)
        
        # Plot equity curve
        ax.plot(df.index, df['equity'], linewidth=2, label='Equity Curve')
        
        # Add benchmark if available
        if 'benchmark_curve' in results:
            benchmark_curve = results['benchmark_curve']
            ax.plot(df.index, benchmark_curve, linewidth=1, alpha=0.7, label='Benchmark')
        
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'equity_curve.png', dpi=self.config.report.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_drawdown_chart(self, charts_dir: Path, results: Dict[str, Any]) -> None:
        """Create drawdown chart.
        
        Args:
            charts_dir: Charts directory
            results: Backtest results
        """
        equity_curve = results.get('equity_curve', [])
        timestamps = results.get('timestamps', [])
        
        if not equity_curve or not timestamps:
            return
        
        fig, ax = plt.subplots(figsize=self.config.report.chart_size)
        
        # Calculate drawdown
        equity = pd.Series(equity_curve)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        
        # Plot drawdown
        ax.fill_between(timestamps, drawdown, 0, alpha=0.3, color='red')
        ax.plot(timestamps, drawdown, color='red', linewidth=1)
        
        ax.set_title('Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'drawdown.png', dpi=self.config.report.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_returns_histogram(self, charts_dir: Path, results: Dict[str, Any]) -> None:
        """Create returns histogram.
        
        Args:
            charts_dir: Charts directory
            results: Backtest results
        """
        trade_results = results.get('trade_results', [])
        
        if not trade_results:
            return
        
        fig, ax = plt.subplots(figsize=self.config.report.chart_size)
        
        # Extract returns
        returns = [t.returns for t in trade_results]
        
        # Create histogram
        ax.hist(returns, bins=30, alpha=0.7, edgecolor='black')
        
        # Add statistics
        mean_return = pd.Series(returns).mean()
        std_return = pd.Series(returns).std()
        
        ax.axvline(mean_return, color='red', linestyle='--', label=f'Mean: {mean_return:.2%}')
        ax.axvline(mean_return + std_return, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_return + std_return:.2%}')
        ax.axvline(mean_return - std_return, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_return - std_return:.2%}')
        
        ax.set_title('Trade Returns Distribution')
        ax.set_xlabel('Returns')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'returns_histogram.png', dpi=self.config.report.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_trade_analysis_charts(self, charts_dir: Path, results: Dict[str, Any]) -> None:
        """Create trade analysis charts.
        
        Args:
            charts_dir: Charts directory
            results: Backtest results
        """
        trade_results = results.get('trade_results', [])
        
        if not trade_results:
            return
        
        # P&L over time
        fig, ax = plt.subplots(figsize=self.config.report.chart_size)
        
        # Extract P&L and timestamps
        pnl = [t.pnl for t in trade_results]
        timestamps = [t.exit_timestamp for t in trade_results]
        
        # Plot P&L
        ax.scatter(timestamps, pnl, alpha=0.6)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_title('Trade P&L Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('P&L')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'trade_pnl.png', dpi=self.config.report.dpi, bbox_inches='tight')
        plt.close()
        
        # Win/Loss by symbol
        fig, ax = plt.subplots(figsize=self.config.report.chart_size)
        
        # Group by symbol
        symbol_pnl = {}
        for trade in trade_results:
            symbol = trade.symbol
            if symbol not in symbol_pnl:
                symbol_pnl[symbol] = []
            symbol_pnl[symbol].append(trade.pnl)
        
        # Calculate total P&L per symbol
        symbol_totals = {symbol: sum(pnl) for symbol, pnl in symbol_pnl.items()}
        
        # Sort by P&L
        sorted_symbols = sorted(symbol_totals.items(), key=lambda x: x[1], reverse=True)
        
        # Plot top 20 symbols
        top_symbols = sorted_symbols[:20]
        symbols, pnls = zip(*top_symbols)
        
        colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
        ax.bar(symbols, pnls, color=colors, alpha=0.7)
        
        ax.set_title('P&L by Symbol (Top 20)')
        ax.set_xlabel('Symbol')
        ax.set_ylabel('Total P&L')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'symbol_pnl.png', dpi=self.config.report.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_setup_analysis_charts(self, charts_dir: Path, results: Dict[str, Any]) -> None:
        """Create setup analysis charts.
        
        Args:
            charts_dir: Charts directory
            results: Backtest results
        """
        trade_results = results.get('trade_results', [])
        
        if not trade_results:
            return
        
        # Group by setup
        setup_stats = {}
        for trade in trade_results:
            setup = trade.setup
            if setup not in setup_stats:
                setup_stats[setup] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': 0.0,
                    'returns': []
                }
            
            setup_stats[setup]['total_trades'] += 1
            if trade.is_winner:
                setup_stats[setup]['winning_trades'] += 1
            else:
                setup_stats[setup]['losing_trades'] += 1
            setup_stats[setup]['total_pnl'] += trade.pnl
            setup_stats[setup]['returns'].append(trade.returns)
        
        # Calculate metrics
        setup_metrics = {}
        for setup, stats in setup_stats.items():
            win_rate = stats['winning_trades'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
            avg_return = pd.Series(stats['returns']).mean() if stats['returns'] else 0
            
            setup_metrics[setup] = {
                'total_trades': stats['total_trades'],
                'win_rate': win_rate,
                'avg_return': avg_return,
                'total_pnl': stats['total_pnl']
            }
        
        # Win rate by setup
        fig, ax = plt.subplots(figsize=self.config.report.chart_size)
        
        setups = list(setup_metrics.keys())
        win_rates = [setup_metrics[setup]['win_rate'] for setup in setups]
        
        ax.bar(setups, win_rates, alpha=0.7)
        ax.set_title('Win Rate by Setup')
        ax.set_xlabel('Setup')
        ax.set_ylabel('Win Rate')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'setup_win_rate.png', dpi=self.config.report.dpi, bbox_inches='tight')
        plt.close()
        
        # Total P&L by setup
        fig, ax = plt.subplots(figsize=self.config.report.chart_size)
        
        total_pnls = [setup_metrics[setup]['total_pnl'] for setup in setups]
        colors = ['green' if pnl > 0 else 'red' for pnl in total_pnls]
        
        ax.bar(setups, total_pnls, color=colors, alpha=0.7)
        ax.set_title('Total P&L by Setup')
        ax.set_xlabel('Setup')
        ax.set_ylabel('Total P&L')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'setup_pnl.png', dpi=self.config.report.dpi, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, run_dir: Path, results: Dict[str, Any]) -> Path:
        """Generate HTML report.
        
        Args:
            run_dir: Run directory
            results: Backtest results
            
        Returns:
            Path to HTML report
        """
        # Prepare template data
        template_data = self._prepare_template_data(results)
        
        # Load template
        template = self.jinja_env.get_template('report.html')
        
        # Render template
        html_content = template.render(**template_data)
        
        # Write to file
        report_path = run_dir / 'report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_markdown_report(self, run_dir: Path, results: Dict[str, Any]) -> Path:
        """Generate Markdown report.
        
        Args:
            run_dir: Run directory
            results: Backtest results
            
        Returns:
            Path to Markdown report
        """
        # Prepare template data
        template_data = self._prepare_template_data(results)
        
        # Load template
        template = self.jinja_env.get_template('report.md')
        
        # Render template
        markdown_content = template.render(**template_data)
        
        # Write to file
        report_path = run_dir / 'report.md'
        with open(report_path, 'w') as f:
            f.write(markdown_content)
        
        return report_path
    
    def _generate_json_report(self, run_dir: Path, results: Dict[str, Any]) -> Path:
        """Generate JSON report.
        
        Args:
            run_dir: Run directory
            results: Backtest results
            
        Returns:
            Path to JSON report
        """
        # Convert results to JSON-serializable format
        json_results = self._convert_to_json(results)
        
        # Write to file
        report_path = run_dir / 'report.json'
        with open(report_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        return report_path
    
    def _prepare_template_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for template rendering.
        
        Args:
            results: Backtest results
            
        Returns:
            Dictionary with template data
        """
        metrics = results.get('metrics', {})
        trade_results = results.get('trade_results', [])
        
        # Calculate additional metrics
        total_trades = len(trade_results)
        winning_trades = len([t for t in trade_results if t.is_winner])
        losing_trades = len([t for t in trade_results if t.is_loser])
        
        # Group by setup
        setup_stats = {}
        for trade in trade_results:
            setup = trade.setup
            if setup not in setup_stats:
                setup_stats[setup] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': 0.0
                }
            
            setup_stats[setup]['total_trades'] += 1
            if trade.is_winner:
                setup_stats[setup]['winning_trades'] += 1
            else:
                setup_stats[setup]['losing_trades'] += 1
            setup_stats[setup]['total_pnl'] += trade.pnl
        
        # Calculate win rates
        for setup, stats in setup_stats.items():
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
        
        return {
            'run_id': results.get('run_id', 'unknown'),
            'config': results.get('config', {}),
            'symbols': results.get('symbols', []),
            'start_date': results.get('start_date', ''),
            'end_date': results.get('end_date', ''),
            'setups': results.get('setups', []),
            'metrics': metrics,
            'setup_stats': setup_stats,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'has_charts': self.config.report.include_charts,
        }
    
    def _convert_to_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results to JSON-serializable format.
        
        Args:
            results: Backtest results
            
        Returns:
            JSON-serializable dictionary
        """
        json_results = {}
        
        for key, value in results.items():
            if key == 'trade_results':
                # Convert trade results to dictionaries
                json_results[key] = [
                    {
                        'symbol': t.symbol,
                        'entry_timestamp': t.entry_timestamp.isoformat(),
                        'exit_timestamp': t.exit_timestamp.isoformat(),
                        'direction': t.direction,
                        'entry_price': t.entry_price,
                        'exit_price': t.exit_price,
                        'quantity': t.quantity,
                        'pnl': t.pnl,
                        'returns': t.returns,
                        'duration_days': t.duration_days,
                        'setup': t.setup,
                        'metadata': t.metadata
                    }
                    for t in value
                ]
            elif key == 'timestamps':
                # Convert timestamps to ISO format
                json_results[key] = [t.isoformat() for t in value]
            elif key == 'metrics':
                # Convert metrics to dictionary
                if hasattr(value, '__dict__'):
                    json_results[key] = value.__dict__
                else:
                    json_results[key] = value
            else:
                json_results[key] = value
        
        return json_results

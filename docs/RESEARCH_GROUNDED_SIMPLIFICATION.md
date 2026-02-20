# Research-Grounded Simplification Notes (2026-02-20)

## Goal

Reduce unnecessary rigidity by removing duplicated risk penalties that can suppress returns without adding meaningful safety.

## External Evidence Used

1. Moreira and Muir (2016/2017), "Volatility Managed Portfolios": volatility timing improves risk-adjusted outcomes when applied at the portfolio level.  
   https://www.nber.org/papers/w22208
2. DeMiguel, Garlappi, and Uppal (2009), "Optimal Versus Naive Diversification": complex optimized allocations can underperform simple rules out of sample because of estimation error and turnover.  
   https://academic.oup.com/rfs/article/22/5/1915/1592901
3. Jagannathan and Ma (2002/2003), "Risk Reduction in Large Portfolios": simple constraints can stabilize portfolios by shrinking noisy covariance structure.  
   https://www.nber.org/papers/w8922
4. Garleanu and Pedersen (2009/2013), "Dynamic Trading with Predictable Returns and Transaction Costs": costs favor partial, disciplined adjustment toward targets, not repeated overlapping penalties.  
   https://www.nber.org/papers/w15205
5. Harvey, Liu, and Zhu (2014/2016), "... and the Cross-Section of Expected Returns": finance signals are vulnerable to data-mining and should use conservative model-complexity discipline.  
   https://www.nber.org/papers/w20592

## Internal Mapping to This Pipeline

Current pipeline already has multiple volatility/risk controls:

1. Hard universe volatility cap (`max_screen_volatility`)
2. Baseline screening volatility penalty (`-0.35 * z(vol_60d_ann)`)
3. Inverse-vol portfolio sizing
4. Portfolio-level volatility targeting
5. Drawdown and no-trade-band overlays

In this context, an additional `ret_per_day / vol` transform inside screening is a second-layer per-name volatility penalty on top of existing controls.

## Implemented Simplification

Changed screening ML priority path to use `ret_per_day` directly (cost-aware, horizon-normalized) instead of `ret_per_day / vol`.

File:

- `stock_screener/screening/screener.py`

Validation added:

- `tests/test_scoring.py` includes a case confirming higher `ret_per_day` is not auto-demoted solely by an extra in-score volatility division.

## Expected Effect

1. Less bias toward ultra-low-volatility names when alpha signal is strong.
2. Preserve existing risk controls at the portfolio layer.
3. Lower model-path complexity by removing one overlapping transformation.

## Implemented Simplification (Follow-up)

Removed duplicated turnover suppression between:

1. Unified optimizer turnover penalty (objective-level)
2. Post-optimizer rebalance turnover shrinkage

Change:

- When `unified_optimizer_enabled=True`, rebalance controls keep hysteresis/min-notional guards but disable extra turnover shrinkage.

Files:

- `stock_screener/pipeline/daily.py`
- `tests/test_rebalance_hysteresis_dynamic.py`

Expected effect:

1. Fewer unnecessary under-target allocations after optimization.
2. Keep transaction-discipline guardrails (small-trade suppression) without double-counting costs.
3. Reduce conservatism introduced by stacked penalties while preserving risk controls.

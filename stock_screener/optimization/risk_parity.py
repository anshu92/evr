from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _cap_weights(w: pd.Series, cap: float, *, allow_cash: bool) -> pd.Series:
    cap = float(cap)
    if cap <= 0 or cap >= 1:
        return w / w.sum()
    w = w.copy()
    for _ in range(10):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w.loc[over] = cap
        under = ~over
        if not under.any():
            break
        under_sum = float(w.loc[under].sum())
        if under_sum > 0:
            w.loc[under] = w.loc[under] + (w.loc[under] / under_sum) * excess
        else:
            # Under-cap names all had weight 0 (alpha col zero/NaN). Distribute excess equally to avoid 0/0 → NaN.
            n_under = int(under.sum())
            if n_under > 0:
                w.loc[under] = excess / n_under
    if allow_cash:
        return w
    total = float(w.sum())
    return w / total if total > 0 else w


def compute_inverse_vol_weights(
    features: pd.DataFrame,
    portfolio_size: int,
    weight_cap: float,
    logger,
    *,
    alpha_col: str | None = None,
    alpha_floor: float = 0.0,
) -> pd.DataFrame:
    """Compute inverse-vol (risk parity style) weights from screened features."""

    df = features.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    n = int(portfolio_size)
    if n <= 0:
        n = 20
    df = df.head(n).copy()

    vol = pd.to_numeric(df["vol_60d_ann"], errors="coerce")
    vol = vol.replace([0.0, np.inf, -np.inf], np.nan).dropna()
    df = df.loc[vol.index].copy()
    if df.empty:
        raise RuntimeError("No tickers with valid volatility for weighting")

    inv = 1.0 / vol
    if alpha_col and alpha_col in df.columns:
        alpha = pd.to_numeric(df[alpha_col], errors="coerce").reindex(vol.index)
        alpha = alpha.clip(lower=float(alpha_floor)).fillna(0.0)
        if float(alpha.sum()) > 0:
            inv = inv * alpha
            logger.info("Applied alpha weighting using %s (floor=%s).", alpha_col, alpha_floor)
        else:
            logger.warning("Alpha column %s has no positive values; falling back to pure inverse-vol.", alpha_col)
    inv_sum = float(inv.sum())
    if not np.isfinite(inv_sum) or inv_sum <= 0:
        logger.warning("Inverse-vol sum invalid (sum=%s); falling back to equal weights.", inv_sum)
        inv = pd.Series(1.0, index=inv.index)
        inv_sum = float(inv.sum())
    w = inv / inv_sum
    cap = float(weight_cap)
    allow_cash = False
    if 0 < cap < 1 and cap * len(w) < 1:
        allow_cash = True
        logger.warning(
            "Weight cap %.3f with %s names implies max invested %.1f%%; leaving cash unallocated.",
            cap,
            len(w),
            cap * len(w) * 100.0,
        )
    w = _cap_weights(w, cap, allow_cash=allow_cash)
    w = w.fillna(0.0)
    if w.sum() > 0:
        w = w / w.sum()

    out = df.copy()
    out["weight"] = w
    out = out.sort_values("weight", ascending=False)
    logger.info("Computed weights: %s tickers (cap=%s)", len(out), weight_cap)
    return out


def compute_correlation_aware_weights(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    portfolio_size: int,
    weight_cap: float,
    logger,
) -> pd.DataFrame:
    """Risk parity with correlation adjustment using equal risk contribution."""
    if not SCIPY_AVAILABLE:
        logger.warning("scipy not available, falling back to inverse-vol weights")
        return compute_inverse_vol_weights(features, portfolio_size, weight_cap, logger)
    
    df = features.head(portfolio_size).copy()
    tickers = list(df.index)
    
    # Get returns for covariance estimation
    try:
        # Extract ticker columns from MultiIndex if needed
        if isinstance(prices.columns, pd.MultiIndex):
            ticker_prices = pd.DataFrame({t: prices[(t, "Close")] for t in tickers if (t, "Close") in prices.columns})
        else:
            ticker_prices = prices[tickers]
        
        returns = ticker_prices.pct_change().dropna()
        
        if len(returns) < 60:
            logger.warning("Insufficient price history (%d days) for covariance, using inverse-vol", len(returns))
            return compute_inverse_vol_weights(features, portfolio_size, weight_cap, logger)
        
        # Compute annualized covariance matrix
        cov_matrix = returns.cov() * 252
        
        # Equal risk contribution optimization
        def risk_budget_objective(weights):
            port_var = weights @ cov_matrix @ weights
            if port_var <= 0:
                return 1e10
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / np.sqrt(port_var + 1e-10)
            target = 1.0 / len(weights)
            return np.sum((risk_contrib - target)**2)
        
        # Constraints: weights sum to 1, all positive, respect cap
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, float(weight_cap)) for _ in range(len(tickers))]
        initial_weights = np.ones(len(tickers)) / len(tickers)
        
        result = minimize(
            risk_budget_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6},
        )
        
        if result.success:
            weights_series = pd.Series(result.x, index=tickers)
            df["weight"] = weights_series
            logger.info("Correlation-aware weights computed successfully")
            return df.sort_values("weight", ascending=False)
        else:
            logger.warning("Optimization failed: %s. Using inverse-vol", result.message)
            return compute_inverse_vol_weights(features, portfolio_size, weight_cap, logger)
    
    except Exception as e:
        logger.warning("Correlation weights failed: %s. Using inverse-vol", str(e))
        return compute_inverse_vol_weights(features, portfolio_size, weight_cap, logger)


def apply_confidence_weighting(
    weights_df: pd.DataFrame,
    confidence: pd.Series | None,
    confidence_floor: float = 0.3,
    logger=None,
) -> pd.DataFrame:
    """Adjust position sizes by model confidence."""
    result = weights_df.copy()
    
    if confidence is None:
        if logger:
            logger.info("No confidence data provided, skipping confidence weighting")
        return result
    
    # Merge confidence if not already in DataFrame
    if "pred_confidence" not in result.columns:
        result = result.join(confidence.rename("pred_confidence"), how="left")
    
    if "pred_confidence" not in result.columns or result["pred_confidence"].isna().all():
        if logger:
            logger.warning("No confidence data available, skipping confidence weighting")
        return result
    
    # Clip confidence to floor
    adj_confidence = result["pred_confidence"].fillna(confidence_floor).clip(lower=confidence_floor)
    
    # Adjust weights by confidence
    result["weight"] = result["weight"] * adj_confidence
    
    # Renormalize
    weight_sum = result["weight"].sum()
    if weight_sum > 0:
        result["weight"] = result["weight"] / weight_sum
    
    if logger:
        logger.info("Applied confidence weighting (floor=%.2f)", confidence_floor)
    
    return result.sort_values("weight", ascending=False)


def apply_conviction_sizing(
    weights_df: pd.DataFrame,
    features: pd.DataFrame,
    *,
    pred_col: str = "pred_return",
    confidence_col: str = "pred_confidence",
    vol_col: str = "vol_60d_ann",
    min_weight_scalar: float = 0.5,
    max_weight_scalar: float = 2.0,
    kelly_fraction: float = 0.25,
    logger=None,
) -> pd.DataFrame:
    """Apply conviction-based position sizing using prediction strength and risk.
    
    This implements a simplified Kelly-inspired approach:
    - Higher predicted returns get larger positions
    - Higher confidence gets larger positions  
    - Higher volatility gets smaller positions (risk-adjusted)
    
    Args:
        weights_df: DataFrame with 'weight' column (base weights)
        features: Features DataFrame with predictions and volatility
        pred_col: Column with calibrated predicted returns
        confidence_col: Column with model confidence
        vol_col: Column with annualized volatility
        min_weight_scalar: Minimum scaling factor (default 0.5x)
        max_weight_scalar: Maximum scaling factor (default 2.0x)
        kelly_fraction: Fraction of Kelly criterion to use (default 0.25 = quarter Kelly)
    """
    result = weights_df.copy()
    
    # Get tickers in common
    common = result.index.intersection(features.index)
    if len(common) == 0:
        if logger:
            logger.warning("No common tickers for conviction sizing")
        return result
    
    # Extract signals
    has_pred = pred_col in features.columns
    has_conf = confidence_col in features.columns
    has_vol = vol_col in features.columns
    
    if not has_pred:
        if logger:
            logger.info("No predictions for conviction sizing, using base weights")
        return result
    
    # Start with neutral scalar
    conviction_scalar = pd.Series(1.0, index=common)
    
    # Factor 1: Predicted return strength (normalized to z-score)
    preds = features.loc[common, pred_col].fillna(0)
    pred_mean = preds.mean()
    pred_std = preds.std()
    if pred_std > 0:
        pred_z = (preds - pred_mean) / pred_std
        # Map z-score to multiplier: z=0 -> 1.0, z=1 -> 1.3, z=-1 -> 0.7
        pred_factor = 1.0 + 0.3 * pred_z.clip(-2, 2)
    else:
        pred_factor = 1.0
    conviction_scalar = conviction_scalar * pred_factor
    
    # Factor 2: Model confidence (if available)
    if has_conf:
        conf = features.loc[common, confidence_col].fillna(0.5)
        # Map confidence to multiplier: conf=0.5 -> 1.0, conf=1.0 -> 1.5, conf=0.3 -> 0.6
        conf_factor = 0.2 + 1.6 * conf.clip(0.2, 1.0)
        conviction_scalar = conviction_scalar * conf_factor
    
    # Factor 3: Risk adjustment via volatility (Kelly-inspired)
    if has_vol:
        vol = features.loc[common, vol_col].fillna(0.30)
        # Higher vol -> smaller position
        # Kelly: f = edge / (odds * variance)
        # Simplified: scalar = kelly_fraction * pred / vol^2, clamped
        vol_clamped = vol.clip(0.10, 0.80)  # Clamp extreme vols
        risk_factor = 0.20 / vol_clamped  # 20% vol is "normal"
        risk_factor = risk_factor.clip(0.5, 1.5)  # Don't go too extreme
        conviction_scalar = conviction_scalar * risk_factor
    
    # Clamp to min/max
    conviction_scalar = conviction_scalar.clip(min_weight_scalar, max_weight_scalar)
    
    # Apply to weights
    for ticker in common:
        if ticker in result.index:
            result.loc[ticker, "weight"] = result.loc[ticker, "weight"] * conviction_scalar[ticker]
    
    # Renormalize
    weight_sum = result["weight"].sum()
    if weight_sum > 0:
        result["weight"] = result["weight"] / weight_sum
    
    if logger:
        avg_scalar = conviction_scalar.mean()
        min_scalar = conviction_scalar.min()
        max_scalar = conviction_scalar.max()
        logger.info(
            "Conviction sizing: avg=%.2fx, range=[%.2f, %.2f]",
            avg_scalar, min_scalar, max_scalar
        )
    
    return result.sort_values("weight", ascending=False)


def compute_volatility_scalar(
    prices: pd.DataFrame,
    target_vol: float = 0.15,
    lookback_days: int = 20,
    min_scalar: float = 0.3,
    max_scalar: float = 1.5,
    logger=None,
) -> float:
    """Compute portfolio exposure scalar based on recent market volatility.
    
    When market volatility is high, reduce exposure; when low, increase it.
    This maintains more consistent risk across different market regimes.
    
    Args:
        prices: DataFrame with price history (expects 'Close' or market prices)
        target_vol: Target annualized volatility (default 15%)
        lookback_days: Days to estimate recent volatility
        min_scalar: Minimum exposure (0.3 = 30% invested)
        max_scalar: Maximum exposure (1.5 = 150% leverage, typically capped at 1.0)
        logger: Logger instance
    """
    try:
        # Try to get a market proxy (cap-weighted average of all stocks)
        if isinstance(prices.columns, pd.MultiIndex):
            # Get all Close prices
            close_cols = [c for c in prices.columns if c[1] == "Close"]
            if not close_cols:
                return 1.0
            close_prices = prices[close_cols].copy()
            close_prices.columns = [c[0] for c in close_prices.columns]
        else:
            close_prices = prices.copy()
        
        if close_prices.empty or len(close_prices) < lookback_days + 5:
            return 1.0
        
        # Equal-weighted market return as proxy (simple approach)
        returns = close_prices.pct_change().dropna()
        market_returns = returns.mean(axis=1)  # Equal-weighted average
        
        if len(market_returns) < lookback_days:
            return 1.0
        
        # Recent realized volatility (annualized)
        recent_vol = float(market_returns.tail(lookback_days).std() * np.sqrt(252))
        
        if recent_vol <= 0 or not np.isfinite(recent_vol):
            return 1.0
        
        # Compute scalar: target_vol / realized_vol
        # High vol → lower scalar; Low vol → higher scalar
        scalar = target_vol / recent_vol
        
        # Clip to bounds
        scalar = np.clip(scalar, min_scalar, max_scalar)
        
        if logger:
            regime = "high" if scalar < 0.8 else ("low" if scalar > 1.2 else "normal")
            logger.info(
                "Volatility targeting: realized_vol=%.1f%%, target=%.1f%%, scalar=%.2f (%s vol regime)",
                recent_vol * 100, target_vol * 100, scalar, regime
            )
        
        return float(scalar)
    
    except Exception as e:
        if logger:
            logger.warning("Volatility scalar failed: %s, using 1.0", e)
        return 1.0


def compute_regime_exposure_scalar(
    features: pd.DataFrame,
    *,
    trend_weight: float = 0.4,
    breadth_weight: float = 0.3,
    vol_weight: float = 0.3,
    min_scalar: float = 0.5,
    max_scalar: float = 1.2,
    logger=None,
) -> float:
    """Compute exposure scalar based on market regime signals.
    
    Uses market trend, breadth, and volatility regime to determine
    whether to be more defensive (scalar < 1) or aggressive (scalar > 1).
    
    Args:
        features: DataFrame with market regime columns
        trend_weight: Weight for market trend signal
        breadth_weight: Weight for market breadth signal
        vol_weight: Weight for inverse volatility signal
        min_scalar: Minimum exposure scalar (defensive)
        max_scalar: Maximum exposure scalar (aggressive)
    """
    # Extract regime signals (use first row - they're same for all stocks)
    if features.empty:
        return 1.0
    
    row = features.iloc[0]
    
    # Trend signal: positive trend -> bullish (score > 0.5)
    # market_trend_20d typically ranges -0.15 to +0.15
    trend = row.get("market_trend_20d", 0.0)
    if pd.isna(trend):
        trend = 0.0
    trend_score = 0.5 + (trend / 0.10) * 0.5  # Map to 0-1 scale
    trend_score = max(0.0, min(1.0, trend_score))
    
    # Breadth signal: >0.5 is bullish, <0.5 is bearish
    breadth = row.get("market_breadth", 0.5)
    if pd.isna(breadth):
        breadth = 0.5
    breadth_score = breadth  # Already 0-1
    
    # Volatility signal: low vol is bullish, high vol is bearish
    # market_vol_regime = 1.0 is normal, >1 is high vol
    vol_regime = row.get("market_vol_regime", 1.0)
    if pd.isna(vol_regime):
        vol_regime = 1.0
    # Invert: low vol (0.8) -> high score (0.6), high vol (1.5) -> low score (0.33)
    vol_score = 1.0 / max(0.5, vol_regime)
    vol_score = max(0.0, min(1.0, vol_score))
    
    # Weighted composite score (0-1 range)
    total_weight = trend_weight + breadth_weight + vol_weight
    composite = (
        trend_weight * trend_score +
        breadth_weight * breadth_score +
        vol_weight * vol_score
    ) / total_weight
    
    # Map composite (0-1) to scalar range (min_scalar to max_scalar)
    # 0.5 -> 1.0, 0.0 -> min_scalar, 1.0 -> max_scalar
    if composite >= 0.5:
        # Bullish: 0.5-1.0 maps to 1.0-max_scalar
        scalar = 1.0 + (composite - 0.5) * 2 * (max_scalar - 1.0)
    else:
        # Bearish: 0.0-0.5 maps to min_scalar-1.0
        scalar = min_scalar + (composite * 2) * (1.0 - min_scalar)
    
    if logger:
        logger.info(
            "Regime exposure: trend=%.2f (%.2f), breadth=%.2f (%.2f), vol=%.2f (%.2f) -> composite=%.2f, scalar=%.2f",
            trend, trend_score, breadth, breadth_score, vol_regime, vol_score, composite, scalar
        )
    
    return scalar


def apply_regime_exposure(
    weights_df: pd.DataFrame,
    features: pd.DataFrame,
    *,
    enabled: bool = True,
    trend_weight: float = 0.4,
    breadth_weight: float = 0.3,
    vol_weight: float = 0.3,
    min_scalar: float = 0.5,
    max_scalar: float = 1.2,
    logger=None,
) -> tuple[pd.DataFrame, float, dict]:
    """Apply regime-aware exposure scaling to portfolio weights.
    
    Returns:
        Tuple of (adjusted weights, cash fraction, regime info dict)
    """
    result = weights_df.copy()
    
    if not enabled or result.empty:
        return result, 0.0, {"enabled": False, "scalar": 1.0}
    
    scalar = compute_regime_exposure_scalar(
        features,
        trend_weight=trend_weight,
        breadth_weight=breadth_weight,
        vol_weight=vol_weight,
        min_scalar=min_scalar,
        max_scalar=max_scalar,
        logger=logger,
    )
    
    # Scale weights
    result["weight"] = result["weight"] * scalar
    
    # Compute cash fraction (if scalar < 1, we hold cash)
    total_invested = result["weight"].sum()
    cash_fraction = max(0.0, 1.0 - total_invested)
    
    # If scalar > 1, renormalize to 100% (leverage not allowed)
    if total_invested > 1.0:
        result["weight"] = result["weight"] / total_invested
        cash_fraction = 0.0
    
    regime_info = {
        "enabled": True,
        "scalar": scalar,
        "cash_fraction": cash_fraction,
    }
    
    return result, cash_fraction, regime_info


def apply_volatility_targeting(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float = 0.15,
    lookback_days: int = 20,
    min_scalar: float = 0.5,
    max_scalar: float = 1.0,
    logger=None,
) -> tuple[pd.DataFrame, float]:
    """Scale portfolio weights based on market volatility regime.
    
    Returns:
        Tuple of (adjusted weights DataFrame, cash allocation fraction)
    """
    result = weights_df.copy()
    
    scalar = compute_volatility_scalar(
        prices=prices,
        target_vol=target_vol,
        lookback_days=lookback_days,
        min_scalar=min_scalar,
        max_scalar=max_scalar,
        logger=logger,
    )
    
    # Scale all weights
    result["weight"] = result["weight"] * scalar
    
    # Compute cash allocation (1 - total invested)
    total_invested = result["weight"].sum()
    cash_allocation = max(0.0, 1.0 - total_invested)
    
    if logger and cash_allocation > 0.01:
        logger.info("Volatility targeting: %.1f%% invested, %.1f%% cash", 
                   total_invested * 100, cash_allocation * 100)
    
    return result, cash_allocation


def apply_liquidity_adjustment(
    weights_df: pd.DataFrame,
    features: pd.DataFrame,
    *,
    liquidity_col: str = "avg_dollar_volume_cad",
    min_liquidity: float = 100_000,
    target_liquidity: float = 1_000_000,
    max_position_pct_of_volume: float = 0.05,
    portfolio_value: float = 10_000,
    min_weight_scalar: float = 0.3,
    logger=None,
) -> pd.DataFrame:
    """Adjust position sizes based on stock liquidity.
    
    Stocks with lower liquidity get smaller positions to:
    1. Avoid market impact when entering/exiting
    2. Ensure positions can be liquidated within a reasonable timeframe
    
    Args:
        weights_df: DataFrame with 'weight' column (base weights)
        features: Features DataFrame with liquidity data
        liquidity_col: Column with average daily dollar volume
        min_liquidity: Minimum liquidity to include stock (filter out illiquid)
        target_liquidity: Liquidity level for full position size
        max_position_pct_of_volume: Max position as % of daily volume (default 5%)
        portfolio_value: Total portfolio value for position sizing cap
        min_weight_scalar: Minimum allowed scaling factor
    """
    result = weights_df.copy()
    
    # Get tickers in common
    common = result.index.intersection(features.index)
    if len(common) == 0:
        if logger:
            logger.warning("No common tickers for liquidity adjustment")
        return result
    
    if liquidity_col not in features.columns:
        if logger:
            logger.info("No liquidity data available, skipping adjustment")
        return result
    
    liquidity = features.loc[common, liquidity_col].fillna(min_liquidity)
    
    # Filter out stocks below minimum liquidity
    illiquid_mask = liquidity < min_liquidity
    n_illiquid = illiquid_mask.sum()
    
    if n_illiquid > 0 and logger:
        logger.info("Removing %d illiquid stocks (<%s daily volume)", 
                   n_illiquid, f"${min_liquidity:,.0f}")
        for ticker in common[illiquid_mask]:
            if ticker in result.index:
                result.loc[ticker, "weight"] = 0.0
    
    # For remaining stocks, scale by liquidity
    liquid_tickers = common[~illiquid_mask]
    
    if len(liquid_tickers) > 0:
        liq = liquidity.loc[liquid_tickers]
        
        # Approach 1: Scale by liquidity relative to target
        # Stocks with target_liquidity get full weight, less liquid get scaled down
        liq_ratio = liq / target_liquidity
        liq_scalar = liq_ratio.clip(min_weight_scalar, 1.0)
        
        # Approach 2: Also cap by max % of daily volume
        # Max position = daily_volume * max_position_pct
        # Weight cap = max_position / portfolio_value
        max_position = liq * max_position_pct_of_volume
        max_weight = max_position / portfolio_value
        
        for ticker in liquid_tickers:
            if ticker in result.index:
                current_weight = result.loc[ticker, "weight"]
                scaled_weight = current_weight * liq_scalar[ticker]
                capped_weight = min(scaled_weight, max_weight[ticker])
                result.loc[ticker, "weight"] = max(0.0, capped_weight)
    
    # Renormalize
    weight_sum = result["weight"].sum()
    if weight_sum > 0:
        result["weight"] = result["weight"] / weight_sum
    
    if logger and len(liquid_tickers) > 0:
        avg_scalar = liq_scalar.mean()
        min_liq = liquidity.loc[liquid_tickers].min()
        logger.info(
            "Liquidity adjustment: avg_scalar=%.2f, min_vol=$%s, %d stocks adjusted",
            avg_scalar, f"{min_liq:,.0f}", len(liquid_tickers)
        )
    
    return result.sort_values("weight", ascending=False)


def apply_correlation_limits(
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    max_corr_weight: float = 0.25,
    corr_threshold: float = 0.7,
    lookback_days: int = 60,
    logger=None,
) -> pd.DataFrame:
    """Limit combined weight of highly correlated positions.
    
    When two stocks have correlation above threshold, their combined weight
    is capped at max_corr_weight to reduce concentration risk.
    
    Args:
        weights_df: DataFrame with 'weight' column
        prices: Historical price DataFrame for correlation calculation
        max_corr_weight: Maximum combined weight for correlated pair (default 25%)
        corr_threshold: Correlation level to trigger limit (default 0.7)
        lookback_days: Days for correlation calculation
    """
    result = weights_df.copy()
    
    if len(result) < 2:
        return result
    
    # Get tickers with non-zero weights
    active_tickers = result[result["weight"] > 0].index.tolist()
    if len(active_tickers) < 2:
        return result
    
    # Extract close prices for active tickers
    try:
        if isinstance(prices.columns, pd.MultiIndex):
            close_cols = [(t, "Close") for t in active_tickers if (t, "Close") in prices.columns]
            if len(close_cols) < 2:
                if logger:
                    logger.info("Insufficient price data for correlation limits")
                return result
            close_prices = prices[close_cols].tail(lookback_days)
            close_prices.columns = [c[0] for c in close_prices.columns]
        else:
            avail = [t for t in active_tickers if t in prices.columns]
            if len(avail) < 2:
                return result
            close_prices = prices[avail].tail(lookback_days)
        
        if close_prices.empty or len(close_prices) < 20:
            return result
        
        # Compute returns and correlation matrix
        returns = close_prices.pct_change().dropna()
        if len(returns) < 20:
            return result
        
        corr_matrix = returns.corr()
        
        # Find highly correlated pairs
        adjustments_made = 0
        pairs_adjusted = []
        
        for i, ticker1 in enumerate(corr_matrix.columns):
            for ticker2 in corr_matrix.columns[i+1:]:
                corr = corr_matrix.loc[ticker1, ticker2]
                
                if not np.isfinite(corr) or corr < corr_threshold:
                    continue
                
                if ticker1 not in result.index or ticker2 not in result.index:
                    continue
                
                w1 = result.loc[ticker1, "weight"]
                w2 = result.loc[ticker2, "weight"]
                combined = w1 + w2
                
                if combined > max_corr_weight:
                    # Scale down both proportionally
                    scale = max_corr_weight / combined
                    result.loc[ticker1, "weight"] = w1 * scale
                    result.loc[ticker2, "weight"] = w2 * scale
                    adjustments_made += 1
                    pairs_adjusted.append((ticker1, ticker2, corr))
        
        # Renormalize
        weight_sum = result["weight"].sum()
        if weight_sum > 0:
            result["weight"] = result["weight"] / weight_sum
        
        if logger and adjustments_made > 0:
            logger.info(
                "Correlation limits: %d pairs adjusted (threshold=%.1f%%, max_weight=%.1f%%)",
                adjustments_made, corr_threshold * 100, max_corr_weight * 100
            )
            for t1, t2, c in pairs_adjusted[:3]:  # Log first 3
                logger.info("  Corr pair: %s-%s (r=%.2f)", t1, t2, c)
    
    except Exception as e:
        if logger:
            logger.warning("Correlation limits failed: %s", e)
    
    return result.sort_values("weight", ascending=False)


def apply_beta_adjustment(
    weights_df: pd.DataFrame,
    features: pd.DataFrame,
    *,
    beta_col: str = "beta",
    target_beta: float = 1.0,
    min_weight_scalar: float = 0.5,
    max_weight_scalar: float = 2.0,
    logger=None,
) -> pd.DataFrame:
    """Adjust position sizes inversely by beta to maintain portfolio risk.
    
    High-beta stocks get smaller positions, low-beta stocks get larger.
    This helps maintain consistent systematic risk exposure.
    
    Args:
        weights_df: DataFrame with 'weight' column
        features: Features DataFrame with beta column
        beta_col: Column name for beta values
        target_beta: Target portfolio beta (default 1.0)
        min_weight_scalar: Minimum adjustment (e.g., 0.5 = halve weight for 2.0 beta)
        max_weight_scalar: Maximum adjustment (e.g., 2.0 = double weight for 0.5 beta)
    """
    result = weights_df.copy()
    
    if beta_col not in features.columns:
        if logger:
            logger.info("Beta adjustment skipped: '%s' column not found", beta_col)
        return result
    
    # Get beta values for stocks in portfolio
    tickers = result.index.tolist()
    beta_values = {}
    
    for ticker in tickers:
        if ticker in features.index:
            beta = features.loc[ticker, beta_col]
        elif "ticker" in features.columns:
            row = features[features["ticker"] == ticker]
            beta = row[beta_col].iloc[0] if len(row) > 0 else None
        else:
            beta = None
        
        if beta is not None and pd.notna(beta) and float(beta) > 0:
            beta_values[ticker] = float(beta)
        else:
            beta_values[ticker] = 1.0  # Default to market beta
    
    if not beta_values:
        return result
    
    # Calculate beta-adjusted weights
    # Scale inversely: weight_scalar = target_beta / stock_beta
    adjustments = {}
    for ticker, beta in beta_values.items():
        if ticker not in result.index:
            continue
        
        # Scale factor: high beta -> smaller weight, low beta -> larger weight
        raw_scalar = target_beta / beta
        scalar = max(min_weight_scalar, min(max_weight_scalar, raw_scalar))
        
        old_weight = result.loc[ticker, "weight"]
        result.loc[ticker, "weight"] = old_weight * scalar
        
        if abs(scalar - 1.0) > 0.1:  # Log significant adjustments
            adjustments[ticker] = (beta, scalar)
    
    # Renormalize
    weight_sum = result["weight"].sum()
    if weight_sum > 0:
        result["weight"] = result["weight"] / weight_sum
    
    if logger and adjustments:
        avg_beta = sum(b for b, _ in adjustments.values()) / len(adjustments)
        logger.info(
            "Beta adjustment: %d stocks adjusted (avg_beta=%.2f, target=%.2f)",
            len(adjustments), avg_beta, target_beta
        )
    
    return result.sort_values("weight", ascending=False)


def apply_min_position_filter(
    weights_df: pd.DataFrame,
    *,
    min_position_pct: float = 0.02,
    logger=None,
) -> pd.DataFrame:
    """Remove positions below minimum weight threshold.
    
    Avoids "dust" positions that are too small to be meaningful
    but still incur transaction costs.
    
    Args:
        weights_df: DataFrame with 'weight' column
        min_position_pct: Minimum weight to keep (e.g., 0.02 = 2%)
    """
    result = weights_df.copy()
    
    if result.empty or "weight" not in result.columns:
        return result
    
    # Identify positions below minimum
    below_min_mask = result["weight"] < min_position_pct
    removed_count = below_min_mask.sum()
    
    if removed_count > 0:
        removed_tickers = result[below_min_mask].index.tolist()
        
        # Remove small positions
        result = result[~below_min_mask].copy()
        
        # Renormalize
        weight_sum = result["weight"].sum()
        if weight_sum > 0:
            result["weight"] = result["weight"] / weight_sum
        
        if logger:
            logger.info(
                "Min position filter: removed %d positions below %.1f%% (%s)",
                removed_count, min_position_pct * 100,
                ", ".join(removed_tickers[:3]) + ("..." if removed_count > 3 else "")
            )
    
    return result.sort_values("weight", ascending=False)


def apply_max_position_cap(
    weights_df: pd.DataFrame,
    *,
    max_position_pct: float = 0.20,
    logger=None,
) -> pd.DataFrame:
    """Enforce maximum weight for any single position.
    
    This is a final safety check to prevent over-concentration,
    applied after all other sizing adjustments.
    
    Args:
        weights_df: DataFrame with 'weight' column
        max_position_pct: Maximum weight for any position (e.g., 0.20 = 20%)
    """
    result = weights_df.copy()
    
    if result.empty or "weight" not in result.columns:
        return result
    
    capped_count = 0
    excess_weight = 0.0
    
    # Cap any positions above max
    for ticker in result.index:
        weight = result.loc[ticker, "weight"]
        if weight > max_position_pct:
            excess_weight += weight - max_position_pct
            result.loc[ticker, "weight"] = max_position_pct
            capped_count += 1
    
    if capped_count > 0:
        # Redistribute excess to uncapped positions proportionally
        uncapped_mask = result["weight"] < max_position_pct
        uncapped_total = result.loc[uncapped_mask, "weight"].sum()
        
        if uncapped_total > 0:
            # Distribute proportionally
            for ticker in result.index:
                if result.loc[ticker, "weight"] < max_position_pct:
                    share = result.loc[ticker, "weight"] / uncapped_total
                    result.loc[ticker, "weight"] += excess_weight * share
        
        # Re-cap in case redistribution pushed any over
        result["weight"] = result["weight"].clip(upper=max_position_pct)
        
        # Renormalize
        weight_sum = result["weight"].sum()
        if weight_sum > 0:
            result["weight"] = result["weight"] / weight_sum
        
        if logger:
            logger.info(
                "Position cap: %d positions capped at %.0f%%",
                capped_count, max_position_pct * 100
            )
    
    return result.sort_values("weight", ascending=False)

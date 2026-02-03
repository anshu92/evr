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

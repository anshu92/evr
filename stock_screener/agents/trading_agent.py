"""Multi-agent LLM trading analysis, inspired by TradingAgents.

Implements analyst -> bull/bear debate -> risk assessment -> final decision
using the Groq free tier (OpenAI-compatible API). No langchain/langgraph
dependency -- lightweight enough for GitHub Actions.

The agent layer runs AFTER ML screening and BEFORE weight optimization,
providing qualitative analysis on the ~8 final candidates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from stock_screener.agents.config import get_agent_config

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


@dataclass
class AgentDecision:
    """Result of the multi-agent analysis for a single ticker."""
    ticker: str
    rating: str           # BUY | OVERWEIGHT | HOLD | UNDERWEIGHT | SELL
    score: float          # -1.0 (strong sell) to +1.0 (strong buy)
    reasoning: str        # Short summary of why
    bull_thesis: str
    bear_thesis: str
    risk_assessment: str


# -- Agent system prompts -------------------------------------------------

_ANALYST_PROMPT = """You are a senior equity analyst. Analyze this stock for a SHORT-TERM trade (1-5 day holding period).

Ticker: {ticker}
Sector: {sector} | Industry: {industry}

ML Model Signal:
- Predicted Return (5d): {pred_return:.2%}
- Confidence: {confidence:.1%}
- Predicted Peak Day: {peak_days:.1f}

Price & Momentum:
- Price: ${price:.2f} CAD
- 5d Return: {ret_5d:.2%} | 10d: {ret_10d:.2%} | 20d: {ret_20d:.2%} | 60d: {ret_60d:.2%} | 120d: {ret_120d:.2%}
- RSI(14): {rsi:.1f}
- vs 20d MA: {ma20_ratio:.2%} | vs 50d MA: {ma50_ratio:.2%} | vs 200d MA: {ma200_ratio:.2%}
- Drawdown from 60d high: {drawdown_60d:.2%}
- Distance from 52w high: {dist_52w_high:.2%} | 52w low: {dist_52w_low:.2%}

Risk:
- Volatility (20d ann): {vol_20d:.2%} | (60d ann): {vol_60d:.2%}
- Beta: {beta:.2f}

Fundamentals:
- Market Cap: {market_cap}
- Trailing P/E: {trailing_pe} | Forward P/E: {forward_pe}
- Price/Book: {price_to_book}
- Profit Margin: {profit_margins} | ROE: {roe}
- Debt/Equity: {debt_to_equity}
- Revenue Growth: {revenue_growth} | Earnings Growth: {earnings_growth}
- Dividend Yield: {dividend_yield}
- Analyst Consensus: {analyst_consensus} ({num_analysts} analysts)

Insider Activity: {insider_activity}

Recent News Headlines:
{news_headlines}

News Sentiment (VADER): {news_sentiment}

{intraday_section}
Market Conditions:
- Vol Regime: {vol_regime:.2f} (1.0=normal, >1.2=high stress)
- Market Trend (20d): {market_trend:.2%}
- Market Breadth: {breadth:.1%}

Provide a concise 2-3 sentence analysis covering the key opportunity and risk. Be specific about the short-term catalyst or headwind based on the news and data above."""

_BULL_PROMPT = """You are a BULL researcher arguing FOR buying {ticker}.

Analyst report: {analyst_report}

Present the strongest 2-3 bullet points for why this stock will outperform in the next 1-5 trading days. Focus on specific catalysts, momentum patterns, and quantitative support from the ML model. Be concise."""

_BEAR_PROMPT = """You are a BEAR researcher arguing AGAINST buying {ticker}.

Analyst report: {analyst_report}
Bull case: {bull_case}

Counter the bull thesis with 2-3 specific risk factors. Focus on what could go wrong in the next 1-5 days: overextension, sentiment reversal, sector headwinds, vol expansion. Be concise."""

_RISK_PROMPT = """You are a risk manager evaluating {ticker} for portfolio inclusion.

Analyst report: {analyst_report}
Bull case: {bull_case}
Bear case: {bear_case}

Current portfolio: {portfolio_context}

Assess: (1) position sizing risk given current portfolio exposure, (2) correlation with existing holdings, (3) downside scenario magnitude. Keep to 2-3 sentences."""

_PM_PROMPT = """You are a portfolio manager making the FINAL decision on {ticker}.

Analyst report: {analyst_report}
Bull case: {bull_case}
Bear case: {bear_case}
Risk assessment: {risk_assessment}

ML model predicts {pred_return:.2%} return over 5 days with {confidence:.1%} confidence.

Respond with EXACTLY this format (no extra text):
RATING: [BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL]
SCORE: [number from -1.0 to 1.0]
REASON: [one sentence]"""


def _create_client(config: dict):
    """Create a Groq/OpenAI-compatible client."""
    if not _GROQ_AVAILABLE:
        return None
    if not config.get("api_key"):
        return None
    return Groq(
        api_key=config["api_key"],
        timeout=config.get("timeout_seconds", 15),
    )


def _call_llm(client, config: dict, system: str, user: str) -> str:
    """Make a single LLM call. Returns response text or empty string on failure."""
    try:
        resp = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=config.get("temperature", 0.3),
            max_tokens=config.get("max_tokens", 1024),
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return ""


def _parse_pm_response(response: str) -> tuple[str, float, str]:
    """Parse the portfolio manager's structured response."""
    rating = "HOLD"
    score = 0.0
    reason = ""

    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("RATING:"):
            raw = line.split(":", 1)[1].strip().upper()
            if raw in ("BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"):
                rating = raw
        elif line.upper().startswith("SCORE:"):
            try:
                score = max(-1.0, min(1.0, float(line.split(":", 1)[1].strip())))
            except ValueError:
                pass
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    if score == 0.0 and rating != "HOLD":
        score = {"BUY": 0.8, "OVERWEIGHT": 0.4, "UNDERWEIGHT": -0.4, "SELL": -0.8}.get(rating, 0.0)

    return rating, score, reason


def _format_news_sentiment(features: dict) -> str:
    avg = features.get("news_sentiment_avg")
    vol = features.get("news_volume_5d", 0)
    if avg is None or (avg != avg):
        return "No recent news data"
    label = "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral"
    return f"{label} ({avg:.2f}), {int(vol)} articles in 5d"


def _format_news_headlines(features: dict) -> str:
    """Format actual news headlines for the LLM prompt."""
    headlines = features.get("news_headlines", [])
    if not headlines:
        return "No recent news available"
    lines = []
    for i, article in enumerate(headlines[:5], 1):
        title = article.get("title", "").strip()
        publisher = article.get("publisher", "")
        date = article.get("publish_date", "")
        if not title:
            continue
        date_str = f" ({date[:10]})" if date else ""
        pub_str = f" — {publisher}" if publisher else ""
        lines.append(f"  {i}. {title}{pub_str}{date_str}")
    return "\n".join(lines) if lines else "No recent news available"


def _format_insider_activity(features: dict) -> str:
    net = features.get("insider_net_buys_90d")
    ratio = features.get("insider_buy_ratio_90d")
    recency = features.get("insider_activity_recency")
    if net is None or (net != net):
        return "No insider data"
    label = "Net buying" if net > 0 else "Net selling" if net < 0 else "Neutral"
    ratio_str = f", buy ratio {ratio:.0%}" if ratio is not None and ratio == ratio else ""
    recency_str = f", {int(recency)}d ago" if recency is not None and recency == recency else ""
    return f"{label} ({net:+,.0f} shares 90d{ratio_str}{recency_str})"


def _fmt_val(v, fmt: str = ".2f", suffix: str = "") -> str:
    """Format a numeric value, returning 'N/A' for NaN/None."""
    if v is None:
        return "N/A"
    try:
        f = float(v)
        if f != f:  # NaN
            return "N/A"
        return f"{f:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _format_market_cap(features: dict) -> str:
    log_cap = features.get("log_market_cap")
    if log_cap is None or (log_cap != log_cap):
        return "N/A"
    cap = 10 ** float(log_cap)
    if cap >= 1e12:
        return f"${cap/1e12:.1f}T"
    if cap >= 1e9:
        return f"${cap/1e9:.1f}B"
    if cap >= 1e6:
        return f"${cap/1e6:.0f}M"
    return f"${cap:,.0f}"


def _format_intraday_section(features: dict) -> str:
    """Format intraday price action if available."""
    change = features.get("intraday_change")
    if change is None or (change != change):
        return ""
    last = features.get("intraday_last")
    high = features.get("intraday_high")
    low = features.get("intraday_low")
    open_px = features.get("intraday_open_today")
    bars = features.get("bars_today", 0)
    lines = ["Intraday Price Action (live):"]
    if open_px and last:
        lines.append(f"- Today's range: {_fmt_val(open_px)} open → {_fmt_val(last)} last ({change:+.2%})")
    if high and low:
        lines.append(f"- Intraday high/low: {_fmt_val(high)} / {_fmt_val(low)}")
    if bars:
        lines.append(f"- Bars observed: {int(bars)}h")
    return "\n".join(lines)


def _format_analyst_consensus(features: dict) -> str:
    rec = features.get("recommendation_mean")
    if rec is None or (rec != rec):
        return "N/A"
    rec = float(rec)
    # yfinance: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell
    label = "Strong Buy" if rec <= 1.5 else "Buy" if rec <= 2.5 else "Hold" if rec <= 3.5 else "Sell" if rec <= 4.5 else "Strong Sell"
    return f"{label} ({rec:.1f})"


def analyze_ticker(
    ticker: str,
    features: dict[str, Any],
    portfolio_context: str = "No current positions",
    config: dict | None = None,
) -> AgentDecision | None:
    """Run the full multi-agent analysis pipeline for a single ticker."""
    cfg = config or get_agent_config()
    client = _create_client(cfg)
    if client is None:
        return None

    # 1. Analyst report
    analyst_prompt = _ANALYST_PROMPT.format(
        ticker=ticker,
        pred_return=features.get("pred_return", 0),
        confidence=features.get("pred_confidence", 0.5),
        peak_days=features.get("pred_peak_days", 3),
        price=features.get("last_close_cad", 0),
        ret_5d=features.get("ret_5d", 0),
        ret_10d=features.get("ret_10d", 0),
        ret_20d=features.get("ret_20d", 0),
        ret_60d=features.get("ret_60d", 0),
        ret_120d=features.get("ret_120d", 0),
        vol_20d=features.get("vol_20d_ann", 0.2),
        vol_60d=features.get("vol_60d_ann", 0.2),
        rsi=features.get("rsi_14", 50),
        beta=features.get("beta", 1.0),
        ma20_ratio=features.get("ma20_ratio", 0),
        ma50_ratio=features.get("ma50_ratio", 0),
        ma200_ratio=features.get("ma200_ratio", 0),
        drawdown_60d=features.get("drawdown_60d", 0),
        dist_52w_high=features.get("dist_52w_high", 0),
        dist_52w_low=features.get("dist_52w_low", 0),
        sector=features.get("sector", "Unknown"),
        industry=features.get("industry", "Unknown"),
        market_cap=_format_market_cap(features),
        trailing_pe=_fmt_val(features.get("trailing_pe")),
        forward_pe=_fmt_val(features.get("forward_pe")),
        price_to_book=_fmt_val(features.get("price_to_book")),
        profit_margins=_fmt_val(features.get("profit_margins"), ".1%"),
        roe=_fmt_val(features.get("return_on_equity"), ".1%"),
        debt_to_equity=_fmt_val(features.get("debt_to_equity")),
        revenue_growth=_fmt_val(features.get("revenue_growth"), ".1%"),
        earnings_growth=_fmt_val(features.get("earnings_growth"), ".1%"),
        dividend_yield=_fmt_val(features.get("dividend_yield"), ".2%"),
        analyst_consensus=_format_analyst_consensus(features),
        num_analysts=_fmt_val(features.get("num_analyst_opinions"), ".0f"),
        news_headlines=_format_news_headlines(features),
        news_sentiment=_format_news_sentiment(features),
        intraday_section=_format_intraday_section(features),
        insider_activity=_format_insider_activity(features),
        vol_regime=features.get("market_vol_regime", 1.0),
        market_trend=features.get("market_trend_20d", 0),
        breadth=features.get("market_breadth", 0.5),
    )
    analyst_report = _call_llm(client, cfg, "You are a senior equity analyst.", analyst_prompt)
    if not analyst_report:
        return None

    # 2. Bull/bear debate
    bull_case = _call_llm(
        client, cfg, "You are a bullish equity researcher.",
        _BULL_PROMPT.format(ticker=ticker, analyst_report=analyst_report),
    )
    bear_case = _call_llm(
        client, cfg, "You are a bearish equity researcher.",
        _BEAR_PROMPT.format(ticker=ticker, analyst_report=analyst_report, bull_case=bull_case),
    )

    # 3. Risk assessment
    risk_assessment = _call_llm(
        client, cfg, "You are a portfolio risk manager.",
        _RISK_PROMPT.format(
            ticker=ticker, analyst_report=analyst_report,
            bull_case=bull_case, bear_case=bear_case,
            portfolio_context=portfolio_context,
        ),
    )

    # 4. Portfolio manager final decision
    pm_response = _call_llm(
        client, cfg, "You are a portfolio manager making trading decisions.",
        _PM_PROMPT.format(
            ticker=ticker, analyst_report=analyst_report,
            bull_case=bull_case, bear_case=bear_case,
            risk_assessment=risk_assessment,
            pred_return=features.get("pred_return", 0),
            confidence=features.get("pred_confidence", 0.5),
        ),
    )

    rating, score, reason = _parse_pm_response(pm_response)

    return AgentDecision(
        ticker=ticker, rating=rating, score=score, reasoning=reason,
        bull_thesis=bull_case, bear_thesis=bear_case, risk_assessment=risk_assessment,
    )


def analyze_candidates(
    candidates: list[dict[str, Any]],
    portfolio_context: str = "No current positions",
    config: dict | None = None,
    log: logging.Logger | None = None,
) -> dict[str, AgentDecision]:
    """Analyze multiple candidates sequentially.

    Returns dict mapping ticker -> AgentDecision.
    """
    _log = log or logger
    cfg = config or get_agent_config()

    if not cfg.get("api_key"):
        _log.info("LLM agent: no API key configured; skipping analysis")
        return {}

    if not _GROQ_AVAILABLE:
        _log.warning("LLM agent: groq package not installed; skipping")
        return {}

    results: dict[str, AgentDecision] = {}
    for c in candidates:
        ticker = c.get("ticker", "")
        if not ticker:
            continue
        try:
            decision = analyze_ticker(ticker, c, portfolio_context=portfolio_context, config=cfg)
            if decision:
                results[ticker] = decision
                _log.info(
                    "LLM agent: %s -> %s (score=%.2f) -- %s",
                    ticker, decision.rating, decision.score, decision.reasoning[:80],
                )
        except Exception as e:
            _log.warning("LLM agent: %s analysis failed: %s", ticker, e)

    _log.info("LLM agent: analyzed %d/%d candidates", len(results), len(candidates))
    return results


def blend_llm_scores(
    features_df,
    decisions: dict[str, AgentDecision],
    score_col: str = "score",
    ml_weight: float = 0.7,
    llm_weight: float = 0.3,
    log: logging.Logger | None = None,
):
    """Blend LLM agent scores into the existing ML score column.

    Returns modified DataFrame with blended scores + llm_rating/llm_score columns.
    """
    import pandas as pd
    import numpy as np

    _log = log or logger

    if not decisions:
        features_df["llm_rating"] = ""
        features_df["llm_score"] = float("nan")
        return features_df

    llm_scores = pd.Series(
        {t: d.score for t, d in decisions.items()}, dtype=float, name="llm_score",
    )
    llm_ratings = pd.Series(
        {t: d.rating for t, d in decisions.items()}, dtype=str, name="llm_rating",
    )

    features_df["llm_score"] = llm_scores.reindex(features_df.index)
    features_df["llm_rating"] = llm_ratings.reindex(features_df.index).fillna("")

    if score_col in features_df.columns:
        has_llm = features_df["llm_score"].notna()
        if has_llm.any():
            ml_z = features_df[score_col]
            llm_z = features_df["llm_score"]
            ml_std = ml_z.std()
            if ml_std > 0:
                llm_z = llm_z * ml_std

            blended = ml_weight * ml_z + llm_weight * llm_z
            features_df.loc[has_llm, score_col] = blended[has_llm]

            _log.info(
                "Blended LLM scores for %d tickers (%.0f%% ML + %.0f%% LLM)",
                has_llm.sum(), ml_weight * 100, llm_weight * 100,
            )

    return features_df

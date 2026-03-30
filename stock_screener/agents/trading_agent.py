"""Multi-agent LLM trading analysis, inspired by TradingAgents.

Implements analyst -> bull/bear debate -> risk assessment -> final decision
using the Groq free tier (OpenAI-compatible API). No langchain/langgraph
dependency -- lightweight enough for GitHub Actions.

The agent layer runs AFTER ML screening and BEFORE weight optimization,
providing qualitative analysis on the ~8 final candidates.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from stock_screener.agents.config import get_agent_config

logger = logging.getLogger(__name__)

# Columns extracted from screened DataFrame for LLM candidate context.
# Shared between daily and intraday pipelines.
_CANDIDATE_COLUMNS: list[str] = [
    "pred_return", "pred_confidence", "pred_peak_days", "score",
    "last_close_cad", "ret_60d", "ret_5d", "ret_10d", "ret_20d", "ret_120d",
    "vol_20d_ann", "vol_60d_ann", "rsi_14",
    "beta", "log_market_cap",
    "ma20_ratio", "ma50_ratio", "ma200_ratio",
    "drawdown_60d", "dist_52w_high", "dist_52w_low",
    "market_vol_regime", "market_trend_20d", "market_breadth",
    "news_sentiment_avg", "news_volume_5d",
    "insider_net_buys_90d", "insider_buy_ratio_90d", "insider_activity_recency",
    "trailing_pe", "forward_pe", "price_to_book",
    "profit_margins", "return_on_equity", "debt_to_equity",
    "revenue_growth", "earnings_growth",
    "dividend_yield", "recommendation_mean", "num_analyst_opinions",
]


def build_agent_candidates(
    screened: pd.DataFrame,
    *,
    max_tickers: int,
    news_by_ticker: dict[str, list] | None = None,
    intraday_context: dict[str, dict] | None = None,
    prices_cad: pd.Series | None = None,
) -> list[dict[str, Any]]:
    """Build the candidate list that feeds into ``analyze_candidates``.

    Consolidates the identical candidate-building logic that was duplicated
    in both ``run_daily`` and ``run_intraday``.
    """
    if news_by_ticker is None:
        news_by_ticker = {}
    if intraday_context is None:
        intraday_context = {}

    tickers = list(screened.index[:max_tickers])
    candidates: list[dict[str, Any]] = []
    for t in tickers:
        row = screened.loc[t]
        candidate: dict[str, Any] = {
            "ticker": str(t),
            **{c: float(row.get(c, float("nan"))) for c in _CANDIDATE_COLUMNS},
            "sector": str(row.get("sector", "Unknown")),
            "industry": str(row.get("industry", "Unknown")),
            "news_headlines": news_by_ticker.get(str(t), []),
        }
        # Override last_close_cad with live price if available
        if prices_cad is not None and str(t) in prices_cad.index:
            candidate["last_close_cad"] = float(prices_cad[str(t)])
        # Merge intraday context (open/high/low/change/bars)
        ic = intraday_context.get(str(t))
        if ic:
            candidate.update(ic)
        candidates.append(candidate)
    return candidates


def build_portfolio_context(
    positions: list,
    cash_cad: float,
    prices_cad: pd.Series | None = None,
) -> str:
    """Build a human-readable portfolio context string for the LLM prompt.

    Consolidates the portfolio context building that was duplicated in both
    ``run_daily`` and ``run_intraday``.
    """
    open_positions = [p for p in positions if getattr(p, "status", "OPEN") == "OPEN"]
    ctx = f"{len(open_positions)} open positions, ${cash_cad:.0f} cash"
    if not open_positions:
        return ctx

    if prices_cad is not None:
        # Intraday path: show P&L per holding
        held_pnl: list[str] = []
        for p in open_positions[:5]:
            px = float(prices_cad.get(p.ticker, float("nan")))
            if pd.notna(px) and getattr(p, "entry_price", 0) > 0:
                pnl = (px / p.entry_price - 1.0) * 100
                held_pnl.append(f"{p.ticker} {pnl:+.1f}%")
            else:
                held_pnl.append(p.ticker)
        ctx += f", holdings: {', '.join(held_pnl)}"
    else:
        # Daily path: just ticker names
        ctx += f", tickers: {', '.join(p.ticker for p in open_positions[:5])}"
    return ctx

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


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
    # Optional enhanced fields (populated when features are enabled)
    debate_rounds: int = 0
    debate_history: list[tuple[str, str]] = field(default_factory=list)
    risk_debate: dict[str, str] | None = None
    analyst_reports: dict[str, str] | None = None
    # Model tracking (which model served the PM decision)
    pm_model: str = ""
    analyst_model: str = ""
    # LLM-primary mode fields (populated when primary_mode=True)
    position_size: str = "MEDIUM"  # SMALL | MEDIUM | FULL | NONE
    target_weight: float | None = None  # 0.0-0.20, LLM-suggested weight
    suggested_stop_loss: float | None = None  # e.g., 0.08
    expected_hold_days: int | None = None  # 1-5


@dataclass
class ExitReviewDecision:
    """LLM recommendation on whether to hold or exit a position."""
    ticker: str
    action: str    # HOLD | TIGHTEN_STOP | EXIT
    urgency: str   # LOW | MEDIUM | HIGH
    reason: str


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

Recent News:
{news_headlines}

Present the strongest 2-3 bullet points for why this stock will outperform in the next 1-5 trading days. Focus on specific catalysts, momentum patterns, and quantitative support from the ML model. Reference specific news headlines if relevant. Be concise."""

_BEAR_PROMPT = """You are a BEAR researcher arguing AGAINST buying {ticker}.

Analyst report: {analyst_report}

Recent News:
{news_headlines}

Present 2-3 specific risk factors for why this stock could UNDERPERFORM in the next 1-5 trading days. Focus on: overextension, sentiment reversal, sector headwinds, vol expansion, negative news catalysts. Be independent — argue from the data, not against another thesis. Be concise."""

_RISK_PROMPT = """You are a risk manager evaluating {ticker} for portfolio inclusion.

Analyst report: {analyst_report}
Bull case: {bull_case}
Bear case: {bear_case}

Recent News:
{news_headlines}

Current portfolio: {portfolio_context}

Assess: (1) position sizing risk given current portfolio exposure, (2) correlation with existing holdings, (3) downside scenario magnitude. Flag any congressional trading activity or unusual news. Keep to 2-3 sentences."""

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

# -- LLM-primary mode prompts (binding decisions) -------------------------

_PM_PROMPT_PRIMARY = """You are a portfolio manager making a BINDING decision on {ticker}. Your output directly determines trading.

Analyst report: {analyst_report}
Bull case: {bull_case}
Bear case: {bear_case}
Risk assessment: {risk_assessment}

ML model predicts {pred_return:.2%} return over 5 days with {confidence:.1%} confidence.
Current portfolio: {portfolio_context}

Respond with EXACTLY this format (no extra text):
RATING: [BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL]
SCORE: [number from -1.0 to 1.0]
POSITION_SIZE: [SMALL|MEDIUM|FULL|NONE]
TARGET_WEIGHT: [decimal 0.0 to 0.20, your suggested portfolio weight]
STOP_LOSS: [decimal, e.g. 0.08 for 8% stop]
HOLD_DAYS: [integer 1-5, expected holding period]
REASON: [one sentence]"""

_PORTFOLIO_PROMPT_PRIMARY = """You are a portfolio strategist with BINDING authority over the final trading portfolio.

CANDIDATES (ranked by score):
{candidate_table}

CURRENT PORTFOLIO:
{portfolio_context}

MARKET CONDITIONS:
- Vol Regime: {vol_regime:.2f} (1.0=normal, >1.2=high stress)
- Market Trend (20d): {market_trend:.2%}
- Market Breadth: {breadth:.1%}

CONSTRAINTS: Max {max_positions} positions, 1-5 day hold, ${budget:.0f} CAD
HARD LIMITS (cannot override): Max 20% per position, 8% stop-loss, 10% portfolio drawdown

Your output DIRECTLY determines portfolio weights. Be precise.

Respond with EXACTLY this format:
CONCENTRATION_RISK: [NONE|LOW|HIGH] - [explanation]
CORRELATION_FLAG: [specific correlated pairs or NONE]
REGIME_CHECK: [OK|CAUTION|REDUCE_EXPOSURE] - [explanation]
TICKER_WEIGHTS: [comma-separated ticker:weight pairs, e.g. AAPL:0.15,MSFT:0.12]
EXCLUDED: [comma-separated tickers to exclude, or NONE]
OVERALL: [one sentence portfolio assessment]"""


_client_cache: dict[str, Any] = {}  # provider:api_key -> client instance


def _get_client(provider: str, api_key: str, base_url: str, sdk: str = "", timeout: int = 15):
    """Get or create a client for any provider. Caches clients for reuse."""
    cache_key = f"{provider}:{api_key[:8]}"
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    client = None
    use_sdk = sdk or ("groq" if provider == "groq" else "openai")

    if use_sdk == "groq" and _GROQ_AVAILABLE:
        client = Groq(api_key=api_key, timeout=timeout)
    elif use_sdk == "openai" and _OPENAI_AVAILABLE:
        client = _OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    elif _GROQ_AVAILABLE:
        # Fallback: use Groq SDK for unknown providers
        client = Groq(api_key=api_key, timeout=timeout)

    if client:
        _client_cache[cache_key] = client
    return client


def _create_client(config: dict):
    """Create client for the primary provider (backward compat)."""
    if not config.get("api_key"):
        return None
    return _get_client(
        config.get("provider", "groq"), config["api_key"],
        config.get("base_url", ""), config.get("sdk", ""),
        config.get("timeout_seconds", 15),
    )


def _create_smart_client(config: dict):
    """Create client for the smart provider (backward compat)."""
    if not config.get("smart_api_key"):
        return None
    return _get_client(
        config.get("smart_provider", "groq"), config["smart_api_key"],
        config.get("smart_base_url", ""), config.get("smart_sdk", ""),
        config.get("timeout_seconds", 15),
    )


_last_call_time: float = 0.0
_rate_limited_models: set[str] = set()  # Models that returned 429 this run — skip them

# Track which model actually served each call (for reporting/audit)
_model_usage: dict[str, int] = {}  # model_name -> call_count


def get_model_usage() -> dict[str, int]:
    """Return model usage counters for the current run (for email legend)."""
    return dict(_model_usage)

# Regex to handle <think>...</think> blocks from reasoning models (qwen3-32b etc.)
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>")


def _clean_think_response(raw: str) -> str:
    """Clean LLM response that may contain <think>...</think> reasoning blocks.

    Strategy: extract the content AFTER the think block as the "answer".
    If the answer is too short (model put everything in <think>), use the
    think content itself — it IS the real analysis, just wrapped in tags.
    Handles multiple think blocks, unclosed tags, and nested tags.
    """
    if "<think>" not in raw:
        return raw.strip()

    # Extract what comes AFTER all think blocks
    answer = _THINK_BLOCK_RE.sub("", raw).strip()
    # Also strip any orphaned/unclosed <think> or </think> tags
    answer = _THINK_TAG_RE.sub("", answer).strip()

    # If the answer is substantial, use it (model followed instructions)
    if len(answer) > 50:
        return answer

    # Answer is too short — the real content is inside <think>.
    # Collect ALL think block contents.
    think_parts = _THINK_BLOCK_RE.findall(raw)
    think_content = "\n\n".join(p.strip() for p in think_parts if p.strip())

    # Combine: think reasoning + any answer fragment
    if answer and think_content:
        return f"{think_content}\n\n{answer}"
    return think_content or answer or _THINK_TAG_RE.sub("", raw).strip()


def _call_llm(
    client, config: dict, system: str, user: str,
    *, max_tokens_override: int | None = None, model_override: str | None = None,
) -> str:
    """Make a single LLM call walking a unified quality-ranked model chain.

    Models are ranked by quality regardless of provider. On 429/failure,
    the next model is tried — which may be on a completely different provider.

    Smart calls (model_override set) try the smart chain instead.
    """
    global _last_call_time
    throttle = config.get("throttle_sleep_seconds", 2.0)
    elapsed = time.time() - _last_call_time
    if _last_call_time > 0 and elapsed < throttle:
        time.sleep(throttle - elapsed)
    _last_call_time = time.time()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    max_tok = max_tokens_override or config.get("max_tokens", 1024)
    base_kwargs: dict[str, Any] = {
        "temperature": config.get("temperature", 0.3),
        "max_completion_tokens": max_tok,
    }

    # Pick chain: smart (for model_override calls) or fast
    if model_override:
        chain = config.get("smart_chain", [])
    else:
        chain = config.get("model_chain", [])

    # Fallback: if chain is old-style list[str], wrap in dicts for backward compat
    if chain and isinstance(chain[0], str):
        chain = [{"model": m, "provider": config.get("provider", "groq"),
                  "api_key": config.get("api_key", ""), "base_url": config.get("base_url", ""),
                  "sdk": config.get("sdk", "groq")} for m in chain]

    if not chain:
        logger.warning("No models available in chain")
        return ""

    from stock_screener.agents.config import REASONING_MODELS

    last_error = None
    for entry in chain:
        model = entry["model"]
        if model in _rate_limited_models:
            continue

        # Get or create client for this model's provider
        model_client = _get_client(
            entry.get("provider", "groq"), entry.get("api_key", ""),
            entry.get("base_url", ""), entry.get("sdk", ""),
            config.get("timeout_seconds", 15),
        )
        if not model_client:
            continue

        try:
            call_kwargs = dict(base_kwargs)
            if model in REASONING_MODELS:
                call_kwargs["reasoning_effort"] = "none"
            resp = model_client.chat.completions.create(model=model, messages=messages, **call_kwargs)
            _model_usage[model] = _model_usage.get(model, 0) + 1
            return _clean_think_response(resp.choices[0].message.content)
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "503" in err_str:
                _rate_limited_models.add(model)
                logger.warning("Rate limit on %s [%s]; trying next in chain", model, entry.get("provider"))
                continue
            # Non-rate-limit error — skip this model but try next (different provider might work)
            logger.warning("LLM call failed on %s [%s]: %s", model, entry.get("provider"), e)
            continue

    logger.warning("All %d models in chain exhausted. Last error: %s", len(chain), last_error)
    return ""


def _smart_call(
    smart_client, fast_client, config: dict, system: str, user: str,
    *, max_tokens_override: int | None = None,
) -> str:
    """Route to smart model chain for critical decisions.

    Uses the smart_chain (Gemini → 70b → qwen3) which is quality-ranked.
    Falls back to the fast chain if all smart models fail.
    """
    # Try smart chain first (walks Gemini → Groq 70b → OpenRouter 70b → etc.)
    result = _call_llm(
        smart_client, config, system, user,
        max_tokens_override=max_tokens_override,
        model_override=config.get("smart_model"),  # Triggers smart_chain in _call_llm
    )
    if result:
        return result
    # All smart models failed — fall back to fast chain
    logger.warning("Smart chain exhausted; falling back to fast chain")
    return _call_llm(fast_client, config, system, user, max_tokens_override=max_tokens_override)


def _extract_field(response: str, field: str) -> str | None:
    """Extract a field value from LLM response, case-insensitive, tolerant of formatting."""
    field_upper = field.upper()
    for line in response.split("\n"):
        stripped = line.strip()
        # Handle: "RATING: BUY", "**RATING:** BUY", "rating: buy"
        cleaned = re.sub(r"\*+", "", stripped)  # strip markdown bold
        if cleaned.upper().startswith(field_upper + ":"):
            return cleaned.split(":", 1)[1].strip()
        # Handle: "RATING BUY" (no colon)
        if cleaned.upper().startswith(field_upper + " "):
            rest = cleaned[len(field):].strip().lstrip(":").strip()
            if rest:
                return rest
    return None


def _parse_pm_response(response: str) -> tuple[str, float, str]:
    """Parse the portfolio manager's structured response. Case-insensitive, format-tolerant."""
    _valid_ratings = {"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"}

    raw_rating = _extract_field(response, "RATING")
    rating = "HOLD"
    _parse_ok = False
    if raw_rating:
        # Extract first valid rating word (handles "BUY (strong momentum)")
        for word in raw_rating.upper().split():
            if word in _valid_ratings:
                rating = word
                _parse_ok = True
                break

    score = 0.0
    raw_score = _extract_field(response, "SCORE")
    if raw_score:
        try:
            # Extract first number from the field (handles "0.8 out of 1.0")
            nums = re.findall(r"-?[0-9]+\.?[0-9]*", raw_score)
            if nums:
                score = max(-1.0, min(1.0, float(nums[0])))
                _parse_ok = True
        except (ValueError, IndexError):
            pass

    reason = _extract_field(response, "REASON") or ""

    if not _parse_ok:
        logger.warning("PM parse: no valid RATING/SCORE found in response (%d chars)", len(response))

    if score == 0.0 and rating != "HOLD":
        score = {"BUY": 0.8, "OVERWEIGHT": 0.4, "UNDERWEIGHT": -0.4, "SELL": -0.8}.get(rating, 0.0)

    return rating, score, reason


def _parse_pm_response_primary(response: str) -> tuple[str, float, str, str, float | None, float | None, int | None]:
    """Parse the enhanced PM response for LLM-primary mode. Uses _extract_field for robustness."""
    rating, score, reason = _parse_pm_response(response)
    position_size = "MEDIUM"
    target_weight = None
    stop_loss = None
    hold_days = None

    raw_ps = _extract_field(response, "POSITION_SIZE")
    if raw_ps:
        for word in raw_ps.upper().split():
            if word in ("SMALL", "MEDIUM", "FULL", "NONE"):
                position_size = word
                break

    raw_tw = _extract_field(response, "TARGET_WEIGHT")
    if raw_tw:
        nums = re.findall(r"[0-9]+\.?[0-9]*", raw_tw)
        if nums:
            try:
                target_weight = max(0.0, min(0.20, float(nums[0])))
            except ValueError:
                pass

    raw_sl = _extract_field(response, "STOP_LOSS")
    if raw_sl:
        nums = re.findall(r"[0-9]+\.?[0-9]*", raw_sl)
        if nums:
            try:
                stop_loss = max(0.01, min(0.30, float(nums[0])))
            except ValueError:
                pass

    raw_hd = _extract_field(response, "HOLD_DAYS")
    if raw_hd:
        nums = re.findall(r"[0-9]+", raw_hd)
        if nums:
            try:
                hold_days = max(1, min(10, int(nums[0])))
            except ValueError:
                pass

    return rating, score, reason, position_size, target_weight, stop_loss, hold_days


def _parse_portfolio_response_primary(response: str) -> dict[str, Any]:
    """Parse the enhanced portfolio response for LLM-primary mode."""
    result = _parse_portfolio_response(response)
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("TICKER_WEIGHTS:"):
            raw = line.split(":", 1)[1].strip()
            weights: dict[str, float] = {}
            for pair in raw.split(","):
                pair = pair.strip()
                if ":" in pair:
                    t, w = pair.split(":", 1)
                    try:
                        weights[t.strip().upper()] = max(0.0, min(0.20, float(w.strip())))
                    except ValueError:
                        pass
            result["ticker_weights"] = weights
        elif line.upper().startswith("EXCLUDED:"):
            raw = line.split(":", 1)[1].strip()
            if raw.upper() != "NONE":
                result["excluded"] = [t.strip().upper() for t in raw.split(",") if t.strip()]
            else:
                result["excluded"] = []
    return result


def _format_news_sentiment(features: dict) -> str:
    avg = features.get("news_sentiment_avg")
    vol = features.get("news_volume_5d", 0)
    if avg is None or (avg != avg):
        return "No recent news data"
    label = "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral"
    return f"{label} ({avg:.2f}), {int(vol)} articles in 5d"


def _format_news_headlines(features: dict) -> str:
    """Format news headlines for the LLM prompt, showing source type."""
    headlines = features.get("news_headlines", [])
    if not headlines:
        return "No recent news available"
    lines = []
    for i, article in enumerate(headlines[:8], 1):
        title = article.get("title", "").strip()
        publisher = article.get("publisher", "")
        date = article.get("publish_date", "")
        source_type = article.get("source_type", "")
        if not title:
            continue
        date_str = f" ({date[:10]})" if date else ""
        pub_str = f" — {publisher}" if publisher else ""
        tag = f"[{source_type}] " if source_type else ""
        lines.append(f"  {i}. {tag}{title}{pub_str}{date_str}")
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


# -- Phase 1: Multi-turn debate prompts ------------------------------------

_BULL_REBUTTAL_PROMPT = """You are a BULL researcher arguing FOR buying {ticker} for a 1-5 day trade.
This is round {round_n} of the debate.

Analyst report: {analyst_report}

DEBATE SO FAR:
{debate_history}

The bear researcher just argued:
{bear_latest}

Respond with:
1. Your strongest counter to the bear's most critical point
2. One NEW catalyst or data point not yet discussed
Keep to 3-4 sentences."""

_BEAR_REBUTTAL_PROMPT = """You are a BEAR researcher arguing AGAINST buying {ticker} for a 1-5 day trade.
This is round {round_n} of the debate.

Analyst report: {analyst_report}

DEBATE SO FAR:
{debate_history}

The bull researcher just argued:
{bull_latest}

Respond with:
1. Your strongest counter to the bull's most optimistic claim
2. One NEW risk factor or data point not yet discussed
Keep to 3-4 sentences."""

_BULL_CLOSING_PROMPT = """You are a BULL researcher. After {num_rounds} rounds of debate about {ticker}, provide your FINAL closing statement.

Full debate:
{full_debate}

Summarize in 2-3 bullet points: (1) the single strongest reason to buy, (2) why the bear's best argument is manageable, (3) expected catalyst and timeline."""

_BEAR_CLOSING_PROMPT = """You are a BEAR researcher. After {num_rounds} rounds of debate about {ticker}, provide your FINAL closing statement.

Full debate:
{full_debate}

Summarize in 2-3 bullet points: (1) the single biggest risk, (2) why the bull's best argument is overstated, (3) the most likely adverse scenario in 1-5 days."""

# -- Phase 2: Risk debate prompts -----------------------------------------

_RISK_UNIFIED_PROMPT = """You are a senior risk officer evaluating {ticker} from THREE perspectives.

Analyst report: {analyst_report}
Bull case: {bull_case}
Bear case: {bear_case}
Recent News: {news_headlines}
Current portfolio: {portfolio_context}

Provide your assessment from all three risk perspectives, then synthesize:

AGGRESSIVE: [1-2 sentences arguing for LARGER position: asymmetric upside, bear risks priced in, opportunity cost]
CONSERVATIVE: [1-2 sentences arguing for SMALLER/NO position: max downside, correlation, tail risks]
NEUTRAL: [1-2 sentences balancing both: recommended size SMALL/MEDIUM/FULL, key factor, stop-loss]
SYNTHESIS: [2-3 sentences: final position size recommendation, key risk to monitor, stop-loss level]"""

# -- Phase 3: Specialized analyst prompts ----------------------------------

_TECHNICAL_ANALYST_PROMPT = """You are a technical analyst evaluating {ticker} for a 1-5 day trade.

Price & Momentum:
- Price: ${price:.2f} CAD
- Returns: 5d={ret_5d:.2%} | 10d={ret_10d:.2%} | 20d={ret_20d:.2%}
- RSI(14): {rsi:.1f}
- vs MAs: 20d={ma20_ratio:.2%} | 50d={ma50_ratio:.2%} | 200d={ma200_ratio:.2%}
- Drawdown from 60d high: {drawdown_60d:.2%}
- 52w range: high={dist_52w_high:.2%} | low={dist_52w_low:.2%}

Volatility: 20d={vol_20d:.2%} | 60d={vol_60d:.2%} | Beta={beta:.2f}

{intraday_section}

ML Model: predicts {pred_return:.2%} return (confidence {confidence:.1%}), peak day {peak_days:.1f}

Market: Vol regime={vol_regime:.2f} | Trend={market_trend:.2%} | Breadth={breadth:.1%}

Respond with EXACTLY this format:
SIGNAL: [BUY|NEUTRAL|SELL]
CONVICTION: [HIGH|MEDIUM|LOW]
KEY_LEVEL: [price level to watch]
ANALYSIS: [2 sentences on momentum/pattern setup]"""

_FUNDAMENTAL_ANALYST_PROMPT = """You are a fundamental analyst evaluating {ticker} ({sector}/{industry}) for a 1-5 day trade.

Valuation:
- Market Cap: {market_cap}
- Trailing P/E: {trailing_pe} | Forward P/E: {forward_pe} | P/B: {price_to_book}

Quality:
- Profit Margin: {profit_margins} | ROE: {roe}
- Debt/Equity: {debt_to_equity}

Growth:
- Revenue Growth: {revenue_growth} | Earnings Growth: {earnings_growth}
- Dividend Yield: {dividend_yield}

Analyst Consensus: {analyst_consensus} ({num_analysts} analysts)

Respond with EXACTLY this format:
SIGNAL: [BUY|NEUTRAL|SELL]
CONVICTION: [HIGH|MEDIUM|LOW]
KEY_METRIC: [the single most important fundamental metric]
ANALYSIS: [2 sentences on valuation/growth setup]"""

_SENTIMENT_ANALYST_PROMPT = """You are a sentiment analyst evaluating {ticker} for a 1-5 day trade.

Recent News Headlines:
{news_headlines}

News Sentiment (VADER): {news_sentiment}

Insider Activity: {insider_activity}

Respond with EXACTLY this format:
SIGNAL: [BUY|NEUTRAL|SELL]
CONVICTION: [HIGH|MEDIUM|LOW]
KEY_CATALYST: [the single most impactful news/insider signal]
ANALYSIS: [2 sentences on sentiment/catalyst setup]"""

_RESEARCH_MANAGER_PROMPT = """You are a research manager synthesizing three analyst reports for {ticker}.

TECHNICAL ANALYST:
{technical_report}

FUNDAMENTAL ANALYST:
{fundamental_report}

SENTIMENT ANALYST:
{sentiment_report}

Synthesize into a 3-4 sentence analysis that weighs each analyst's signal. Note any disagreements between analysts. Identify the single most actionable insight for a 1-5 day trade."""

# -- Phase 4: Portfolio reasoning prompt -----------------------------------

_PORTFOLIO_PROMPT = """You are a portfolio strategist reviewing the final candidate list for a short-term trading portfolio.

CANDIDATES (ranked by blended ML+LLM score):
{candidate_table}

CURRENT PORTFOLIO:
{portfolio_context}

MARKET CONDITIONS:
- Vol Regime: {vol_regime:.2f} (1.0=normal, >1.2=high stress)
- Market Trend (20d): {market_trend:.2%}
- Market Breadth: {breadth:.1%}

CONSTRAINTS: Max {max_positions} positions, 1-5 day hold, ${budget:.0f} CAD

Review for:
1. SECTOR CONCENTRATION: Flag if >40% weight in one sector
2. CORRELATION RISK: Flag if multiple candidates are in the same industry
3. REGIME MISMATCH: Flag aggressive picks in a high-vol regime
4. WEIGHT ADJUSTMENTS: Suggest specific weight changes

Respond with EXACTLY this format:
CONCENTRATION_RISK: [NONE|LOW|HIGH] - [explanation]
CORRELATION_FLAG: [correlated pairs or NONE]
REGIME_CHECK: [OK|CAUTION|REDUCE_EXPOSURE] - [explanation]
ADJUSTMENTS: [specific ticker weight changes or NONE]
OVERALL: [one sentence portfolio assessment]"""

# -- Phase 5: Exit review prompt -------------------------------------------

_EXIT_REVIEW_PROMPT = """You are an exit strategy specialist reviewing whether to HOLD or EXIT {ticker}.

POSITION:
- Entry price: ${entry_price:.2f} CAD | Current: ${current_price:.2f} CAD
- Unrealized P&L: {pnl_pct:+.2%}
- Days held: {days_held} of {max_hold} max (remaining: {remaining_days} trading days)
- Entry ML prediction: {entry_pred_return:.2%} return over 5 days

CURRENT SIGNALS:
- Current ML predicted return: {current_pred_return:.2%} (vs {entry_pred_return:.2%} at entry)
- Signal change: {'IMPROVING' if current_pred_return > entry_pred_return else 'DETERIORATING' if current_pred_return < entry_pred_return else 'STABLE'}
- RSI(14): {rsi:.1f}
- Today's price change: {intraday_change:+.2%}
- Drawdown from peak since entry: {peak_drawdown:.2%}

RECENT NEWS:
{recent_news}

Market: Vol regime={vol_regime:.2f}, trend={market_trend:.2%}

Mechanical exit rules (stop-loss, trailing stop, max hold) say HOLD. Consider whether news, signal deterioration, or market conditions warrant overriding.

Respond with EXACTLY this format:
ACTION: [HOLD|TIGHTEN_STOP|EXIT]
URGENCY: [LOW|MEDIUM|HIGH]
REASON: [one sentence]"""


# ── Phase 1: Multi-turn debate engine ─────────────────────────────────────

def _format_debate_history(history: list[tuple[str, str]]) -> str:
    """Format debate history for inclusion in prompts."""
    if not history:
        return "(no prior debate)"
    lines = []
    for i, (side, text) in enumerate(history, 1):
        lines.append(f"[Round {(i + 1) // 2} — {side}]: {text}")
    return "\n".join(lines)


def _run_debate(
    client, cfg: dict, ticker: str, analyst_report: str, news_headlines: str = "",
) -> tuple[str, str, list[tuple[str, str]]]:
    """Run multi-turn bull/bear debate. Returns (bull_closing, bear_closing, history)."""
    max_rounds = max(1, cfg.get("max_debate_rounds", 1))
    debate_tokens = cfg.get("debate_max_tokens", 256)
    history: list[tuple[str, str]] = []
    _nh = news_headlines or "No recent news available"

    # Round 1: independent arguments (no asymmetry — both see only analyst report)
    bull_r1 = _call_llm(
        client, cfg, "You are a bullish equity researcher.",
        _BULL_PROMPT.format(ticker=ticker, analyst_report=analyst_report, news_headlines=_nh),
        max_tokens_override=debate_tokens,
    )
    history.append(("BULL", bull_r1))

    bear_r1 = _call_llm(
        client, cfg, "You are a bearish equity researcher.",
        _BEAR_PROMPT.format(ticker=ticker, analyst_report=analyst_report, news_headlines=_nh),
        max_tokens_override=debate_tokens,
    )
    history.append(("BEAR", bear_r1))

    # Additional rebuttal rounds
    for round_n in range(2, max_rounds + 1):
        history_text = _format_debate_history(history)

        bull_rn = _call_llm(
            client, cfg, "You are a bullish equity researcher.",
            _BULL_REBUTTAL_PROMPT.format(
                ticker=ticker, round_n=round_n, analyst_report=analyst_report,
                debate_history=history_text, bear_latest=history[-1][1],
            ),
            max_tokens_override=debate_tokens,
        )
        history.append(("BULL", bull_rn))

        bear_rn = _call_llm(
            client, cfg, "You are a bearish equity researcher.",
            _BEAR_REBUTTAL_PROMPT.format(
                ticker=ticker, round_n=round_n, analyst_report=analyst_report,
                debate_history=_format_debate_history(history), bull_latest=bull_rn,
            ),
            max_tokens_override=debate_tokens,
        )
        history.append(("BEAR", bear_rn))

    # Closing statements (only for multi-round debates)
    if max_rounds > 1:
        full_debate = _format_debate_history(history)
        bull_closing = _call_llm(
            client, cfg, "You are a bullish equity researcher.",
            _BULL_CLOSING_PROMPT.format(ticker=ticker, num_rounds=max_rounds, full_debate=full_debate),
            max_tokens_override=debate_tokens,
        )
        bear_closing = _call_llm(
            client, cfg, "You are a bearish equity researcher.",
            _BEAR_CLOSING_PROMPT.format(ticker=ticker, num_rounds=max_rounds, full_debate=full_debate),
            max_tokens_override=debate_tokens,
        )
    else:
        bull_closing = bull_r1
        bear_closing = bear_r1

    return bull_closing, bear_closing, history


# ── Phase 2: 3-way risk debate ────────────────────────────────────────────

def _run_risk_debate(
    client, cfg: dict, ticker: str, analyst_report: str,
    bull_case: str, bear_case: str, portfolio_context: str,
    news_headlines: str = "",
) -> tuple[str, dict[str, str]]:
    """Run unified risk assessment (1 call instead of 4). Returns (synthesis, views_dict)."""
    risk_tokens = cfg.get("debate_max_tokens", 256) * 2  # More budget for unified output
    _nh = news_headlines or "No recent news available"

    response = _call_llm(
        client, cfg, "You are a senior risk officer.",
        _RISK_UNIFIED_PROMPT.format(
            ticker=ticker, analyst_report=analyst_report,
            bull_case=bull_case, bear_case=bear_case,
            portfolio_context=portfolio_context, news_headlines=_nh,
        ),
        max_tokens_override=risk_tokens,
    )

    # Parse the unified response into views
    aggressive = _extract_field(response, "AGGRESSIVE") or ""
    conservative = _extract_field(response, "CONSERVATIVE") or ""
    neutral = _extract_field(response, "NEUTRAL") or ""
    synthesis = _extract_field(response, "SYNTHESIS") or response  # Fallback to full response

    views = {"aggressive": aggressive, "conservative": conservative, "neutral": neutral}
    return synthesis, views


# ── Phase 3: Specialized analysts ─────────────────────────────────────────

def _run_specialized_analysts(
    client, cfg: dict, ticker: str, features: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    """Run technical/fundamental/sentiment analysts + research manager synthesis."""
    analyst_tokens = cfg.get("analyst_max_tokens", 200)

    tech_report = _call_llm(
        client, cfg, "You are a technical analyst.",
        _TECHNICAL_ANALYST_PROMPT.format(
            ticker=ticker,
            price=features.get("last_close_cad", 0),
            ret_5d=features.get("ret_5d", 0),
            ret_10d=features.get("ret_10d", 0),
            ret_20d=features.get("ret_20d", 0),
            rsi=features.get("rsi_14", 50),
            ma20_ratio=features.get("ma20_ratio", 0),
            ma50_ratio=features.get("ma50_ratio", 0),
            ma200_ratio=features.get("ma200_ratio", 0),
            drawdown_60d=features.get("drawdown_60d", 0),
            dist_52w_high=features.get("dist_52w_high", 0),
            dist_52w_low=features.get("dist_52w_low", 0),
            vol_20d=features.get("vol_20d_ann", 0.2),
            vol_60d=features.get("vol_60d_ann", 0.2),
            beta=features.get("beta", 1.0),
            intraday_section=_format_intraday_section(features),
            pred_return=features.get("pred_return", 0),
            confidence=features.get("pred_confidence", 0.5),
            peak_days=features.get("pred_peak_days", 3),
            vol_regime=features.get("market_vol_regime", 1.0),
            market_trend=features.get("market_trend_20d", 0),
            breadth=features.get("market_breadth", 0.5),
        ),
        max_tokens_override=analyst_tokens,
    )

    fund_report = _call_llm(
        client, cfg, "You are a fundamental analyst.",
        _FUNDAMENTAL_ANALYST_PROMPT.format(
            ticker=ticker,
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
        ),
        max_tokens_override=analyst_tokens,
    )

    sent_report = _call_llm(
        client, cfg, "You are a sentiment analyst.",
        _SENTIMENT_ANALYST_PROMPT.format(
            ticker=ticker,
            news_headlines=_format_news_headlines(features),
            news_sentiment=_format_news_sentiment(features),
            insider_activity=_format_insider_activity(features),
        ),
        max_tokens_override=analyst_tokens,
    )

    synthesis = _call_llm(
        client, cfg, "You are a research manager.",
        _RESEARCH_MANAGER_PROMPT.format(
            ticker=ticker,
            technical_report=tech_report or "No technical analysis available",
            fundamental_report=fund_report or "No fundamental analysis available",
            sentiment_report=sent_report or "No sentiment analysis available",
        ),
    )

    reports = {"technical": tech_report, "fundamental": fund_report, "sentiment": sent_report}
    return synthesis or "", reports


# ── Core analysis (updated with Phase 1-3) ────────────────────────────────

def analyze_ticker(
    ticker: str,
    features: dict[str, Any],
    portfolio_context: str = "No current positions",
    config: dict | None = None,
    primary_mode: bool = False,
) -> AgentDecision | None:
    """Run the full multi-agent analysis pipeline for a single ticker.

    Uses fast provider (Groq) for analysts/debates, smart provider (Gemini) for PM decision.
    """
    cfg = config or get_agent_config()
    client = _create_client(cfg)
    if client is None:
        return None
    smart_client = _create_smart_client(cfg)

    # 1. Analyst report (generic or specialized) — uses FAST provider
    analyst_reports = None
    if cfg.get("specialized_analysts"):
        analyst_report, analyst_reports = _run_specialized_analysts(client, cfg, ticker, features)
    else:
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

    # Enrich analyst report with specialist summaries if available
    _enriched_report = analyst_report
    if isinstance(analyst_reports, dict):
        _specialist_lines = []
        for key, label in [("technical", "Technical"), ("fundamental", "Fundamental"), ("sentiment", "Sentiment")]:
            rpt = analyst_reports.get(key, "")
            if rpt:
                _specialist_lines.append(f"[{label}] {rpt[:150]}")
        if _specialist_lines:
            _enriched_report = analyst_report + "\n\nSpecialist Signals:\n" + "\n".join(_specialist_lines)

    # Format news for debate/risk prompts (all agents see the same headlines)
    _news_fmt = _format_news_headlines(features)

    # 2. Bull/bear debate — uses enriched report so specialists feed into debate
    bull_case, bear_case, debate_history = _run_debate(
        client, cfg, ticker, _enriched_report, news_headlines=_news_fmt,
    )

    # 3. Risk assessment (Phase 2: 3-way debate when enabled)
    risk_debate = None
    if cfg.get("risk_debate_enabled"):
        risk_assessment, risk_debate = _run_risk_debate(
            client, cfg, ticker, analyst_report,
            bull_case, bear_case, portfolio_context,
            news_headlines=_news_fmt,
        )
    else:
        risk_assessment = _call_llm(
            client, cfg, "You are a portfolio risk manager.",
            _RISK_PROMPT.format(
                ticker=ticker, analyst_report=analyst_report,
                bull_case=bull_case, bear_case=bear_case,
                portfolio_context=portfolio_context,
                news_headlines=_news_fmt,
            ),
        )

    # 4. Portfolio manager final decision — uses SMART provider (Gemini)
    position_size = "MEDIUM"
    target_weight = None
    suggested_stop_loss = None
    expected_hold_days = None

    if primary_mode:
        pm_response = _smart_call(
            smart_client, client, cfg,
            "You are a portfolio manager making binding trading decisions.",
            _PM_PROMPT_PRIMARY.format(
                ticker=ticker, analyst_report=analyst_report,
                bull_case=bull_case, bear_case=bear_case,
                risk_assessment=risk_assessment,
                pred_return=features.get("pred_return", 0),
                confidence=features.get("pred_confidence", 0.5),
                portfolio_context=portfolio_context,
            ),
        )
        rating, score, reason, position_size, target_weight, suggested_stop_loss, expected_hold_days = (
            _parse_pm_response_primary(pm_response)
        )
    else:
        pm_response = _smart_call(
            smart_client, client, cfg,
            "You are a portfolio manager making trading decisions.",
            _PM_PROMPT.format(
                ticker=ticker, analyst_report=analyst_report,
                bull_case=bull_case, bear_case=bear_case,
                risk_assessment=risk_assessment,
                pred_return=features.get("pred_return", 0),
                confidence=features.get("pred_confidence", 0.5),
            ),
        )
        rating, score, reason = _parse_pm_response(pm_response)

    # Determine which models were used
    _pm_model = cfg.get("smart_model", cfg["model"]) if smart_client else cfg["model"]
    _analyst_model = cfg["model"]  # Analysts always use fast provider

    return AgentDecision(
        ticker=ticker, rating=rating, score=score, reasoning=reason,
        bull_thesis=bull_case, bear_thesis=bear_case, risk_assessment=risk_assessment,
        debate_rounds=cfg.get("max_debate_rounds", 1),
        debate_history=debate_history,
        risk_debate=risk_debate,
        analyst_reports=analyst_reports,
        pm_model=_pm_model,
        analyst_model=_analyst_model,
        position_size=position_size,
        target_weight=target_weight,
        suggested_stop_loss=suggested_stop_loss,
        expected_hold_days=expected_hold_days,
    )


def analyze_candidates(
    candidates: list[dict[str, Any]],
    portfolio_context: str = "No current positions",
    config: dict | None = None,
    log: logging.Logger | None = None,
    primary_mode: bool = False,
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
            decision = analyze_ticker(ticker, c, portfolio_context=portfolio_context, config=cfg, primary_mode=primary_mode)
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


# ── Phase 4: Portfolio-level LLM reasoning ────────────────────────────────

def _parse_portfolio_response(response: str) -> dict[str, str]:
    """Parse the portfolio reasoner's structured response."""
    result: dict[str, str] = {}
    for line in response.split("\n"):
        line = line.strip()
        for key in ("CONCENTRATION_RISK", "CORRELATION_FLAG", "REGIME_CHECK", "ADJUSTMENTS", "OVERALL"):
            if line.upper().startswith(key + ":"):
                result[key.lower()] = line.split(":", 1)[1].strip()
                break
    return result


def analyze_portfolio(
    candidates: list[dict[str, Any]],
    decisions: dict[str, AgentDecision],
    portfolio_context: str,
    market_conditions: dict[str, Any],
    max_positions: int = 8,
    budget_cad: float = 500.0,
    config: dict | None = None,
    log: logging.Logger | None = None,
    primary_mode: bool = False,
) -> dict[str, Any] | None:
    """Run portfolio-level LLM reasoning (Phase 4). Uses smart provider (Gemini)."""
    _log = log or logger
    cfg = config or get_agent_config()
    client = _create_client(cfg)
    if client is None:
        return None
    smart_client = _create_smart_client(cfg)

    table_lines = []
    for c in candidates:
        t = c.get("ticker", "?")
        d = decisions.get(t)
        sector = c.get("sector", "?")
        pred = c.get("pred_return", 0)
        if d:
            table_lines.append(
                f"- {t} ({sector}): LLM={d.rating}({d.score:+.1f}), "
                f"ML_pred={pred:.2%}, vol={c.get('vol_20d_ann', 0):.1%}"
            )

    if not table_lines:
        return None

    prompt_template = _PORTFOLIO_PROMPT_PRIMARY if primary_mode else _PORTFOLIO_PROMPT
    response = _smart_call(
        smart_client, client, cfg,
        "You are a portfolio strategist with binding authority." if primary_mode else "You are a portfolio strategist.",
        prompt_template.format(
            candidate_table="\n".join(table_lines),
            portfolio_context=portfolio_context,
            vol_regime=market_conditions.get("vol_regime", 1.0),
            market_trend=market_conditions.get("market_trend", 0),
            breadth=market_conditions.get("breadth", 0.5),
            max_positions=max_positions,
            budget=budget_cad,
        ),
        max_tokens_override=cfg.get("portfolio_max_tokens", 512),
    )

    if not response:
        return None

    result = _parse_portfolio_response_primary(response) if primary_mode else _parse_portfolio_response(response)
    result["raw_response"] = response
    _log.info("Portfolio reasoning: %s", result.get("overall", "(no overall)"))
    return result


# ── Phase 5: Exit/Hold LLM review ────────────────────────────────────────

def _parse_exit_review(response: str) -> tuple[str, str, str]:
    """Parse exit review response. Returns (action, urgency, reason)."""
    action = "HOLD"
    urgency = "LOW"
    reason = ""
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("ACTION:"):
            raw = line.split(":", 1)[1].strip().upper()
            if raw in ("HOLD", "TIGHTEN_STOP", "EXIT"):
                action = raw
        elif line.upper().startswith("URGENCY:"):
            raw = line.split(":", 1)[1].strip().upper()
            if raw in ("LOW", "MEDIUM", "HIGH"):
                urgency = raw
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return action, urgency, reason


def review_exits(
    held_positions: list,
    prices_cad: pd.Series,
    features: pd.DataFrame | None,
    news_by_ticker: dict[str, list] | None = None,
    market_conditions: dict[str, Any] | None = None,
    max_hold_days: int = 10,
    config: dict | None = None,
    log: logging.Logger | None = None,
) -> dict[str, ExitReviewDecision]:
    """Review held positions for potential LLM-recommended exits (Phase 5). Uses smart provider."""
    _log = log or logger
    cfg = config or get_agent_config()
    client = _create_client(cfg)
    if client is None:
        return {}
    smart_client = _create_smart_client(cfg)
    if news_by_ticker is None:
        news_by_ticker = {}
    if market_conditions is None:
        market_conditions = {}

    exit_tokens = cfg.get("exit_review_max_tokens", 256)
    results: dict[str, ExitReviewDecision] = {}

    for pos in held_positions:
        ticker = getattr(pos, "ticker", "")
        if not ticker:
            continue

        current_price = float(prices_cad.get(ticker, 0))
        entry_price = float(getattr(pos, "entry_price", 0))
        if entry_price <= 0 or current_price <= 0:
            continue

        pnl_pct = (current_price / entry_price) - 1.0
        days_held = 0
        if hasattr(pos, "entry_date"):
            from datetime import datetime, timezone
            now = datetime.now(tz=timezone.utc)
            diff = (now - pos.entry_date).days
            days_held = max(0, diff)

        # Get current ML predictions if available
        current_pred = 0.0
        rsi = 50.0
        intraday_change = 0.0
        peak_drawdown = 0.0
        if features is not None and not features.empty and ticker in features.index:
            row = features.loc[ticker]
            current_pred = float(row.get("pred_return", 0))
            rsi = float(row.get("rsi_14", 50))
            if "intraday_change" in features.columns:
                intraday_change = float(row.get("intraday_change", 0))
            peak_drawdown = float(row.get("drawdown_60d", 0))

        # Format news
        news = news_by_ticker.get(ticker, [])
        news_text = "No recent news"
        if news:
            news_lines = [f"- {a.get('title', '')}" for a in news[:3] if a.get("title")]
            if news_lines:
                news_text = "\n".join(news_lines)

        try:
            response = _smart_call(
                smart_client, client, cfg, "You are an exit strategy specialist.",
                _EXIT_REVIEW_PROMPT.format(
                    ticker=ticker,
                    entry_price=entry_price,
                    current_price=current_price,
                    pnl_pct=pnl_pct,
                    days_held=days_held,
                    max_hold=max_hold_days,
                    remaining_days=max(0, max_hold_days - days_held),
                    entry_pred_return=float(getattr(pos, "pred_return", 0) or 0),
                    current_pred_return=current_pred,
                    rsi=rsi,
                    intraday_change=intraday_change,
                    peak_drawdown=peak_drawdown,
                    recent_news=news_text,
                    vol_regime=market_conditions.get("vol_regime", 1.0),
                    market_trend=market_conditions.get("market_trend", 0),
                ),
                max_tokens_override=exit_tokens,
            )

            if response:
                action, urgency, reason = _parse_exit_review(response)
                results[ticker] = ExitReviewDecision(
                    ticker=ticker, action=action, urgency=urgency, reason=reason,
                )
                _log.info("Exit review: %s -> %s (urgency=%s) -- %s", ticker, action, urgency, reason[:60])
        except Exception as e:
            _log.warning("Exit review failed for %s: %s", ticker, e)

    return results


# ── LLM-Primary Decision Functions ────────────────────────────────────────

def select_tickers_llm_primary(
    screened: pd.DataFrame,
    decisions: dict[str, AgentDecision],
    max_positions: int,
    log: logging.Logger | None = None,
) -> list[str]:
    """Select tickers based on LLM ratings (LLM-primary mode).

    BUY/OVERWEIGHT → include (sorted by LLM score).
    HOLD → fill remaining slots (ML score as tiebreaker).
    UNDERWEIGHT/SELL → exclude.
    """
    _log = log or logger
    buy_tickers: list[tuple[str, float]] = []
    hold_tickers: list[str] = []
    excluded: list[str] = []

    for t, d in decisions.items():
        if d.rating in ("BUY", "OVERWEIGHT"):
            buy_tickers.append((t, d.score))
        elif d.rating == "HOLD":
            hold_tickers.append(t)
        else:
            excluded.append(t)

    buy_tickers.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in buy_tickers]

    # Fill remaining slots with HOLD tickers using ML score as tiebreaker
    if len(selected) < max_positions and hold_tickers:
        hold_scored: list[tuple[str, float]] = []
        for t in hold_tickers:
            ml = float(screened.loc[t, "score"]) if t in screened.index and "score" in screened.columns else 0.0
            hold_scored.append((t, ml))
        hold_scored.sort(key=lambda x: x[1], reverse=True)
        for t, _ in hold_scored:
            if len(selected) >= max_positions:
                break
            selected.append(t)

    final = selected[:max_positions]
    _log.info(
        "LLM-primary ticker selection: %d BUY/OW, %d HOLD fill, %d excluded → %d selected",
        len(buy_tickers), len(final) - min(len(buy_tickers), max_positions),
        len(excluded), len(final),
    )
    return final


def compute_llm_primary_weights(
    selected_tickers: list[str],
    decisions: dict[str, AgentDecision],
    screened: pd.DataFrame,
    *,
    small_range: tuple[float, float] = (0.05, 0.10),
    medium_range: tuple[float, float] = (0.10, 0.15),
    full_range: tuple[float, float] = (0.15, 0.20),
    max_position_pct: float = 0.20,
    min_position_pct: float = 0.02,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Compute portfolio weights from LLM position sizing decisions."""
    _log = log or logger
    size_ranges = {
        "SMALL": small_range,
        "MEDIUM": medium_range,
        "FULL": full_range,
        "NONE": (0.0, 0.0),
    }

    raw_weights: dict[str, float] = {}
    for t in selected_tickers:
        d = decisions.get(t)
        if d is None:
            raw_weights[t] = sum(medium_range) / 2
            continue

        # Use explicit target_weight if provided
        if d.target_weight is not None and d.target_weight > 0:
            raw_weights[t] = min(d.target_weight, max_position_pct)
            continue

        # Map position_size to range, interpolate using LLM score
        lo, hi = size_ranges.get(d.position_size, medium_range)
        if lo == 0.0 and hi == 0.0:
            continue
        alpha = (d.score + 1.0) / 2.0  # [-1,1] → [0,1]
        raw_weights[t] = lo + alpha * (hi - lo)

    total = sum(raw_weights.values())
    if total <= 0:
        _log.warning("LLM weights sum to zero; falling back to equal weights")
        n = len(selected_tickers)
        if n > 0:
            raw_weights = {t: 1.0 / n for t in selected_tickers}
            total = 1.0
        else:
            return pd.DataFrame({"weight": pd.Series(dtype=float)})

    weights = {t: w / total for t, w in raw_weights.items()}

    # Apply hard caps
    for t in weights:
        weights[t] = max(min_position_pct, min(max_position_pct, weights[t]))

    # Re-normalize after caps
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    result = screened.loc[screened.index.isin(selected_tickers)].copy()
    result["weight"] = pd.Series(weights)
    result = result.dropna(subset=["weight"])
    result = result.sort_values("weight", ascending=False)

    _log.info("LLM-primary weights: %s", {t: f"{w:.1%}" for t, w in weights.items()})
    return result


def apply_portfolio_reasoning_enforced(
    target_weights: pd.DataFrame,
    portfolio_reasoning: dict[str, Any],
    screened: pd.DataFrame,
    *,
    sector_cap: float = 0.30,
    regime_reduce_scalar: float = 0.50,
    max_position_pct: float = 0.20,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Apply portfolio-level LLM reasoning as BINDING adjustments."""
    _log = log or logger
    tw = target_weights.copy()

    # 1. Apply explicit ticker weights from LLM (case-insensitive match)
    ticker_weights = portfolio_reasoning.get("ticker_weights", {})
    if isinstance(ticker_weights, dict) and ticker_weights:
        for t, w in ticker_weights.items():
            mask = tw.index.str.upper() == str(t).upper()
            if mask.any():
                tw.loc[mask, "weight"] = min(float(w), max_position_pct)
        _log.info("Enforced LLM ticker weights: %s", ticker_weights)

    # 2. Exclude tickers flagged by LLM
    excluded = portfolio_reasoning.get("excluded", [])
    if isinstance(excluded, list) and excluded:
        before = len(tw)
        tw = tw[~tw.index.str.upper().isin([e.upper() for e in excluded])]
        _log.info("Excluded %d tickers per LLM: %s", before - len(tw), excluded)

    # 3. Concentration risk: cap sector weights
    conc = str(portfolio_reasoning.get("concentration_risk", "")).upper()
    if "HIGH" in conc and "sector" in screened.columns and not tw.empty:
        sector_groups: dict[str, list[str]] = {}
        for t in tw.index:
            s = str(screened.loc[t, "sector"]) if t in screened.index else "Unknown"
            sector_groups.setdefault(s, []).append(t)
        for sector, tickers in sector_groups.items():
            sector_total = float(tw.loc[tw.index.isin(tickers), "weight"].sum())
            if sector_total > sector_cap:
                scale = sector_cap / sector_total
                tw.loc[tw.index.isin(tickers), "weight"] *= scale
                _log.info("Sector cap enforced: %s %.1f%% → %.1f%%", sector, sector_total * 100, sector_cap * 100)

    # 4. Regime check: reduce exposure
    regime = str(portfolio_reasoning.get("regime_check", "")).upper()
    if "REDUCE_EXPOSURE" in regime:
        tw["weight"] *= regime_reduce_scalar
        _log.info("Regime reduce enforced: all weights scaled by %.0f%%", regime_reduce_scalar * 100)

    # 5. Re-normalize
    wsum = float(tw["weight"].sum()) if not tw.empty else 0.0
    if wsum > 0:
        tw["weight"] /= wsum

    return tw

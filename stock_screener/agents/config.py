"""Configuration for the LLM trading agent layer."""
from __future__ import annotations

import os


def get_agent_config() -> dict:
    """Build agent config from environment variables.

    Uses Groq free tier by default (OpenAI-compatible API).
    Override with AGENT_LLM_PROVIDER, AGENT_LLM_MODEL, etc.
    """
    provider = os.getenv("AGENT_LLM_PROVIDER", "groq").lower()

    # Provider-specific defaults
    defaults = {
        "groq": {
            "api_key_env": "GROQ_API_KEY",
            "base_url": "https://api.groq.com/openai/v1",
            "model": "llama-3.3-70b-versatile",
        },
        "openrouter": {
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "meta-llama/llama-3.1-8b-instruct:free",
        },
        "openai": {
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        },
    }

    cfg = defaults.get(provider, defaults["groq"])

    return {
        "provider": provider,
        "api_key": os.getenv(cfg["api_key_env"], ""),
        "base_url": os.getenv("AGENT_LLM_BASE_URL", cfg["base_url"]),
        "model": os.getenv("AGENT_LLM_MODEL", cfg["model"]),
        "temperature": float(os.getenv("AGENT_LLM_TEMPERATURE", "0.3")),
        "max_tokens": int(os.getenv("AGENT_LLM_MAX_TOKENS", "1024")),
        "timeout_seconds": int(os.getenv("AGENT_LLM_TIMEOUT", "15")),
        # Debate configuration
        "max_debate_rounds": int(os.getenv("AGENT_MAX_DEBATE_ROUNDS", "2")),
        "max_risk_rounds": int(os.getenv("AGENT_MAX_RISK_ROUNDS", "1")),
        "debate_max_tokens": int(os.getenv("AGENT_DEBATE_MAX_TOKENS", "256")),
        "analyst_max_tokens": int(os.getenv("AGENT_ANALYST_MAX_TOKENS", "200")),
        "portfolio_max_tokens": int(os.getenv("AGENT_PORTFOLIO_MAX_TOKENS", "512")),
        "exit_review_max_tokens": int(os.getenv("AGENT_EXIT_MAX_TOKENS", "256")),
        # Feature toggles
        "specialized_analysts": os.getenv("AGENT_SPECIALIZED_ANALYSTS", "1").lower() in ("1", "true"),
        "risk_debate_enabled": os.getenv("AGENT_RISK_DEBATE_ENABLED", "1").lower() in ("1", "true"),
        "portfolio_reasoning": os.getenv("AGENT_PORTFOLIO_REASONING", "1").lower() in ("1", "true"),
        "exit_review_enabled": os.getenv("AGENT_EXIT_REVIEW_ENABLED", "1").lower() in ("1", "true"),
        # Rate limiting
        "throttle_sleep_seconds": float(os.getenv("AGENT_THROTTLE_SLEEP", "2.0")),
        "max_llm_tickers": int(os.getenv("AGENT_MAX_LLM_TICKERS", "8")),
    }

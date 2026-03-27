"""Configuration for the LLM trading agent layer.

Dual-provider architecture:
  - "fast" provider (Groq, default): high-volume per-ticker analysis
    (analysts, bull/bear debate, risk debate rounds)
  - "smart" provider (Gemini, optional): few critical decisions
    (portfolio manager, portfolio reasoning, exit review)

When GEMINI_API_KEY is set, high-stakes calls automatically route to Gemini.
When only GROQ_API_KEY is set, everything uses Groq.
"""
from __future__ import annotations

import os

# Provider presets
_PROVIDERS = {
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",  # 500K+ TPD limit (vs 100K for 70b)
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
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


def _build_provider_config(provider_name: str) -> dict:
    """Build config dict for a single provider."""
    preset = _PROVIDERS.get(provider_name, _PROVIDERS["groq"])
    return {
        "provider": provider_name,
        "api_key": os.getenv(preset["api_key_env"], ""),
        "base_url": os.getenv(f"AGENT_{provider_name.upper()}_BASE_URL", preset["base_url"]),
        "model": os.getenv(f"AGENT_{provider_name.upper()}_MODEL", preset["model"]),
    }


def get_agent_config() -> dict:
    """Build agent config from environment variables.

    Dual-provider: "fast" (Groq) for volume, "smart" (Gemini) for decisions.
    Override primary provider with AGENT_LLM_PROVIDER.
    """
    primary = os.getenv("AGENT_LLM_PROVIDER", "groq").lower()
    smart_provider = os.getenv("AGENT_SMART_PROVIDER", "gemini").lower()

    primary_cfg = _build_provider_config(primary)
    smart_cfg = _build_provider_config(smart_provider)

    # Smart provider falls back to primary if no API key
    smart_available = bool(smart_cfg.get("api_key"))

    return {
        # Primary (fast) provider — used for analysts, debates, risk rounds
        "provider": primary_cfg["provider"],
        "api_key": primary_cfg["api_key"],
        "base_url": os.getenv("AGENT_LLM_BASE_URL", primary_cfg["base_url"]),
        "model": os.getenv("AGENT_LLM_MODEL", primary_cfg["model"]),
        "temperature": float(os.getenv("AGENT_LLM_TEMPERATURE", "0.3")),
        "max_tokens": int(os.getenv("AGENT_LLM_MAX_TOKENS", "1024")),
        "timeout_seconds": int(os.getenv("AGENT_LLM_TIMEOUT", "15")),
        # Smart provider — used for PM decisions, portfolio reasoning, exit review
        "smart_provider": smart_cfg["provider"] if smart_available else primary_cfg["provider"],
        "smart_api_key": smart_cfg["api_key"] if smart_available else primary_cfg["api_key"],
        "smart_base_url": smart_cfg["base_url"] if smart_available else primary_cfg["base_url"],
        "smart_model": smart_cfg["model"] if smart_available else primary_cfg["model"],
        "smart_available": smart_available,
        # Debate configuration
        "max_debate_rounds": int(os.getenv("AGENT_MAX_DEBATE_ROUNDS", "1")),
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
        "max_llm_tickers": int(os.getenv("AGENT_MAX_LLM_TICKERS", "5")),
    }

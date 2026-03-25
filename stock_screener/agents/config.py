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
        "max_debate_rounds": int(os.getenv("AGENT_MAX_DEBATE_ROUNDS", "1")),
        "max_risk_rounds": int(os.getenv("AGENT_MAX_RISK_ROUNDS", "1")),
    }

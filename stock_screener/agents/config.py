"""Configuration for the LLM trading agent layer.

Unified model ranking: models are ranked by quality regardless of provider.
On 429/failure, the next model in the ranked list is tried automatically.
Each model specifies its provider so the correct client is created.

Ranking (fast calls — analysts, debates, risk):
  1. llama-3.3-70b      (Groq)        — best reasoning, 100K TPD
  2. llama-3.3-70b      (OpenRouter)   — same quality, different quota
  3. qwen3-32b          (Groq)         — strong reasoning, 500K TPD
  4. hermes-3-405b      (OpenRouter)   — massive 405B when available
  5. llama-4-scout-17b  (Groq)         — MoE, 500K TPD
  6. gemma-3-27b        (OpenRouter)   — decent quality
  7. kimi-k2            (Groq)         — good quality, 300K TPD
  8. llama-3.1-8b       (Groq)         — fast last resort

Ranking (smart calls — PM decisions, portfolio, exits):
  1. gemini-2.5-flash   (Gemini)       — best structured output
  Then falls back to the fast chain above.
"""
from __future__ import annotations

import os
from typing import Any


# ── Unified model ranking ─────────────────────────────────────────────────
# Each entry: (model_id, provider, notes)
# Provider determines which API key/base_url to use.

_FAST_MODEL_CHAIN: list[tuple[str, str]] = [
    # Tier 1: Best reasoning (70B+)
    ("llama-3.3-70b-versatile",                     "groq"),        # 70B, 100K TPD Groq
    ("meta-llama/llama-3.3-70b-instruct:free",      "openrouter"),  # 70B, same quality different quota
    ("nousresearch/hermes-3-llama-3.1-405b:free",   "openrouter"),  # 405B, best when available
    ("nvidia/nemotron-3-super-120b-a12b:free",      "openrouter"),  # 120B MoE (12B active), 262K ctx
    ("openai/gpt-oss-120b:free",                    "openrouter"),  # 120B, GPT-class
    # Tier 2: Strong reasoning (30-80B)
    ("qwen/qwen3-32b",                              "groq"),        # 32B, 500K TPD Groq
    ("qwen/qwen3-next-80b-a3b-instruct:free",       "openrouter"),  # 80B MoE (3B active), 262K ctx
    ("qwen/qwen3.6-plus-preview:free",              "openrouter"),  # Latest Qwen, 1M ctx
    ("meta-llama/llama-4-scout-17b-16e-instruct",   "groq"),        # 17B MoE, 500K TPD Groq
    ("stepfun/step-3.5-flash:free",                 "openrouter"),  # 256K ctx
    ("z-ai/glm-4.5-air:free",                       "openrouter"),  # 131K ctx, GLM family
    ("arcee-ai/trinity-large-preview:free",          "openrouter"),  # 131K ctx
    # Tier 3: Good quality (9-30B)
    ("moonshotai/kimi-k2-instruct",                 "groq"),        # 300K TPD Groq
    ("google/gemma-3-27b-it:free",                  "openrouter"),  # 27B, 131K ctx
    ("nvidia/nemotron-3-nano-30b-a3b:free",         "openrouter"),  # 30B MoE (3B active), 256K ctx
    ("minimax/minimax-m2.5:free",                   "openrouter"),  # 196K ctx
    ("openai/gpt-oss-20b:free",                     "openrouter"),  # 20B, GPT-class
    ("google/gemma-3-12b-it:free",                  "openrouter"),  # 12B, 32K ctx
    # Tier 4: Fast fallback (small models)
    ("llama-3.1-8b-instant",                        "groq"),        # 8B, 500K TPD, 14K RPD
    ("nvidia/nemotron-nano-9b-v2:free",             "openrouter"),  # 9B, 128K ctx
    ("arcee-ai/trinity-mini:free",                  "openrouter"),  # Small, 131K ctx
]

_SMART_MODEL_CHAIN: list[tuple[str, str]] = [
    # Best structured output + reasoning for PM decisions
    ("gemini-2.5-flash",                            "gemini"),      # Best structured output
    ("llama-3.3-70b-versatile",                     "groq"),        # Strong reasoning
    ("meta-llama/llama-3.3-70b-instruct:free",      "openrouter"),  # Same quality
    ("nousresearch/hermes-3-llama-3.1-405b:free",   "openrouter"),  # 405B massive
    ("nvidia/nemotron-3-super-120b-a12b:free",      "openrouter"),  # 120B
    ("openai/gpt-oss-120b:free",                    "openrouter"),  # 120B GPT-class
    ("qwen/qwen3-32b",                              "groq"),        # Strong reasoning
    ("qwen/qwen3.6-plus-preview:free",              "openrouter"),  # Latest Qwen
]

# Provider connection presets
_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "sdk": "groq",  # Use Groq SDK (don't pass base_url)
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "sdk": "openai",  # Use OpenAI SDK
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "sdk": "openai",  # Use OpenAI SDK
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "sdk": "openai",
    },
}

# Models that support reasoning_effort="none" to disable <think> blocks
REASONING_MODELS: set[str] = {
    "qwen/qwen3-32b",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "qwen/qwen3.6-plus-preview:free",
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-120b", "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b", "openai/gpt-oss-20b:free",
}


def _get_available_providers() -> dict[str, str]:
    """Return {provider_name: api_key} for providers with valid API keys."""
    available: dict[str, str] = {}
    for name, preset in _PROVIDER_PRESETS.items():
        key = os.getenv(preset["api_key_env"], "").strip()
        if key:
            available[name] = key
    return available


def _build_model_chain(
    chain: list[tuple[str, str]],
    available_providers: dict[str, str],
) -> list[dict[str, Any]]:
    """Build resolved model chain, filtering out models whose provider has no API key."""
    result: list[dict[str, Any]] = []
    for model_id, provider in chain:
        if provider not in available_providers:
            continue
        preset = _PROVIDER_PRESETS[provider]
        result.append({
            "model": model_id,
            "provider": provider,
            "api_key": available_providers[provider],
            "base_url": preset["base_url"],
            "sdk": preset["sdk"],
        })
    return result


def get_agent_config() -> dict:
    """Build agent config from environment variables.

    Models ranked by quality, not grouped by provider.
    """
    available = _get_available_providers()

    fast_chain = _build_model_chain(_FAST_MODEL_CHAIN, available)
    smart_chain = _build_model_chain(_SMART_MODEL_CHAIN, available)

    # Primary model = first available in fast chain
    primary = fast_chain[0] if fast_chain else {"model": "", "provider": "", "api_key": "", "base_url": "", "sdk": "groq"}
    # Smart model = first available in smart chain
    smart = smart_chain[0] if smart_chain else primary

    return {
        # Primary model + full ranked chain
        "provider": primary["provider"],
        "api_key": primary["api_key"],
        "base_url": primary["base_url"],
        "sdk": primary["sdk"],
        "model": primary["model"],
        "model_chain": fast_chain,  # Full ranked list with provider info
        "temperature": float(os.getenv("AGENT_LLM_TEMPERATURE", "0.3")),
        "max_tokens": int(os.getenv("AGENT_LLM_MAX_TOKENS", "1024")),
        "timeout_seconds": int(os.getenv("AGENT_LLM_TIMEOUT", "15")),
        # Smart model + chain (for PM decisions, portfolio, exits)
        "smart_provider": smart["provider"],
        "smart_api_key": smart["api_key"],
        "smart_base_url": smart["base_url"],
        "smart_sdk": smart["sdk"],
        "smart_model": smart["model"],
        "smart_chain": smart_chain,
        "smart_available": bool(smart_chain),
        # Available providers (for client creation)
        "available_providers": available,
        # Debate configuration
        "max_debate_rounds": int(os.getenv("AGENT_MAX_DEBATE_ROUNDS", "1")),
        "max_risk_rounds": int(os.getenv("AGENT_MAX_RISK_ROUNDS", "1")),
        "debate_max_tokens": int(os.getenv("AGENT_DEBATE_MAX_TOKENS", "400")),
        "analyst_max_tokens": int(os.getenv("AGENT_ANALYST_MAX_TOKENS", "400")),
        "portfolio_max_tokens": int(os.getenv("AGENT_PORTFOLIO_MAX_TOKENS", "600")),
        "exit_review_max_tokens": int(os.getenv("AGENT_EXIT_MAX_TOKENS", "400")),
        # Feature toggles
        "specialized_analysts": os.getenv("AGENT_SPECIALIZED_ANALYSTS", "1").lower() in ("1", "true"),
        "risk_debate_enabled": os.getenv("AGENT_RISK_DEBATE_ENABLED", "1").lower() in ("1", "true"),
        "portfolio_reasoning": os.getenv("AGENT_PORTFOLIO_REASONING", "1").lower() in ("1", "true"),
        "exit_review_enabled": os.getenv("AGENT_EXIT_REVIEW_ENABLED", "1").lower() in ("1", "true"),
        # Rate limiting
        "throttle_sleep_seconds": float(os.getenv("AGENT_THROTTLE_SLEEP", "2.0")),
        "max_llm_tickers": int(os.getenv("AGENT_MAX_LLM_TICKERS", "5")),
    }

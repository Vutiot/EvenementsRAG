"""
OpenAI-compatible client factory for the supported LLM providers.

All providers we call (NVIDIA build, OpenRouter, OpenAI) speak the OpenAI
chat-completions protocol, so one ``openai.OpenAI`` client per provider is
enough. API keys are always read from settings / environment variables,
never from config files.

Usage:
    from src.llm import get_llm_client, extract_usage, usage_request_kwargs

    client = get_llm_client("openrouter")
    resp = client.chat.completions.create(
        model="z-ai/glm-5.3-flash", messages=[...], **usage_request_kwargs("openrouter")
    )
    usage = extract_usage(resp)   # {"prompt_tokens": .., "completion_tokens": .., "cost_usd": ..}
"""

from functools import lru_cache
from typing import Optional

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _provider_credentials(provider: str) -> tuple[str, Optional[str]]:
    """Return ``(api_key, base_url)`` for *provider*."""
    if provider == "openrouter":
        return settings.OPENROUTER_API_KEY, settings.OPENROUTER_BASE_URL
    if provider == "openai":
        return settings.OPENAI_API_KEY, None
    if provider == "nvidia":
        return settings.NVIDIA_API_KEY, settings.NVIDIA_BASE_URL
    raise ValueError(
        f"Unsupported LLM provider '{provider}' (expected openrouter, nvidia or openai)"
    )


@lru_cache(maxsize=8)
def get_llm_client(provider: str):
    """Return a cached ``openai.OpenAI`` client configured for *provider*."""
    import openai

    api_key, base_url = _provider_credentials(provider)
    if not api_key:
        logger.warning(
            f"No API key set for provider '{provider}'. LLM calls will fail. "
            f"Set {provider.upper()}_API_KEY in the environment or .env file."
        )
    kwargs = {"api_key": api_key or "missing-key"}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def usage_request_kwargs(provider: Optional[str]) -> dict:
    """Extra request kwargs so the provider reports token usage and cost.

    OpenRouter adds ``usage.cost`` (USD) to the response when usage
    accounting is requested; other providers already return token counts.
    """
    if provider == "openrouter":
        return {"extra_body": {"usage": {"include": True}}}
    return {}


def extract_usage(response) -> dict:
    """Extract token counts (and OpenRouter cost when present) from a response.

    Returns a dict with ``prompt_tokens``, ``completion_tokens`` and
    ``cost_usd`` (``None`` when the provider does not report a cost).
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "cost_usd": None}

    def _get(name):
        if isinstance(usage, dict):
            return usage.get(name)
        val = getattr(usage, name, None)
        if val is None:
            extra = getattr(usage, "model_extra", None) or {}
            val = extra.get(name)
        return val

    cost = _get("cost")
    return {
        "prompt_tokens": _get("prompt_tokens"),
        "completion_tokens": _get("completion_tokens"),
        "cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
    }

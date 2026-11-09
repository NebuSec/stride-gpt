"""Unified model registry — single source of truth for providers and models.

To add or update a model, edit the MODELS list below. No other files need to change.
Last reviewed: 2026-09-26

Provider model listing page:
  - OpenAI: https://developers.openai.com/api/docs/models
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderInfo:
    """Provider-level metadata (auth, routing, UI text)."""

    name: str  # Display name: "OpenAI"
    provider_key: str  # Routing key used in LLMConfig.provider
    env_var: str | None = None
    cli_prefix: str = ""  # e.g. "openai/" for --worker-model openai/gpt-6-sol
    api_key_url: str = ""
    setup_instructions: str = ""


@dataclass(frozen=True)
class ModelInfo:
    """Per-model metadata."""

    model_id: str  # API model identifier
    provider_key: str  # Must match a ProviderInfo.provider_key
    default_tokens: int = 64000
    max_tokens: int = 128000
    uses_max_completion_tokens: bool = False  # reasoning models
    supports_tools: bool = True
    help_text: str = ""


# ---------------------------------------------------------------------------
# Provider registry — keyed by display name
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, ProviderInfo] = {
    "OpenAI": ProviderInfo(
        name="OpenAI",
        provider_key="OpenAI API",
        env_var="OPENAI_API_KEY",
        cli_prefix="openai/",
        api_key_url="https://platform.openai.com/account/api-keys",
        setup_instructions=(
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) "
            "and chosen model below 🔑\n"
            "2. Provide details of the application that you would like to threat model  📝\n"
            "3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀"
        ),
    ),
}


# ---------------------------------------------------------------------------
# Model registry — order determines UI display order per provider
# ---------------------------------------------------------------------------

MODELS: list[ModelInfo] = [
    # --- OpenAI ---
    ModelInfo(
        model_id="gpt-6-astra",
        provider_key="OpenAI API",
        default_tokens=128000,
        max_tokens=1050000,
        uses_max_completion_tokens=True,
        help_text="GPT-6 Astra is OpenAI's most capable model, built for the hardest end-to-end work.",
    ),
    ModelInfo(
        model_id="gpt-6-sol",
        provider_key="OpenAI API",
        default_tokens=128000,
        max_tokens=1050000,
        uses_max_completion_tokens=True,
        help_text="GPT-6 Sol balances intelligence and cost for complex coding and agentic workflows.",
    ),
    ModelInfo(
        model_id="gpt-6-luna",
        provider_key="OpenAI API",
        default_tokens=128000,
        max_tokens=1050000,
        uses_max_completion_tokens=True,
        help_text="GPT-6 Luna is the most efficient option for focused, high-volume tasks.",
    ),
]

# Build lookup indexes at import time
_PROVIDER_BY_KEY: dict[str, ProviderInfo] = {
    p.provider_key: p for p in PROVIDERS.values()
}
_MODEL_LOOKUP: dict[tuple[str, str], ModelInfo] = {
    (m.provider_key, m.model_id): m for m in MODELS
}
_COMPLETION_TOKEN_IDS: set[str] = {
    m.model_id for m in MODELS if m.uses_max_completion_tokens
}


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_models_for_provider(provider_key: str) -> list[ModelInfo]:
    """Return models for a provider, in UI display order."""
    return [m for m in MODELS if m.provider_key == provider_key]


def get_model(provider_key: str, model_id: str) -> ModelInfo | None:
    """Look up a specific model by provider key and model ID."""
    return _MODEL_LOOKUP.get((provider_key, model_id))


def get_provider_by_key(provider_key: str) -> ProviderInfo | None:
    """Look up a provider by its routing key (e.g. 'OpenAI API')."""
    return _PROVIDER_BY_KEY.get(provider_key)


def model_uses_completion_tokens(model_id: str) -> bool:
    """True if this model uses max_completion_tokens / max_output_tokens."""
    return model_id in _COMPLETION_TOKEN_IDS

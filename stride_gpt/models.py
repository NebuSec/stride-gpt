"""Unified model registry — single source of truth for providers and models.

Provider model listing pages:
  - OpenAI:    https://platform.openai.com/docs/models
  - Mistral:   https://docs.mistral.ai/getting-started/models
  - Groq:      https://console.groq.com/docs/models
  - DeepSeek:  https://api-docs.deepseek.com/quick_start/pricing
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderInfo:
    """Provider-level metadata (auth, routing, UI text)."""

    name: str  # Display name: "OpenAI", "Groq", etc.
    provider_key: str  # Routing key used in LLMConfig.provider
    env_var: str | None = None
    needs_api_key: bool = True
    needs_api_base: bool = False
    default_api_base: str | None = None
    # OpenAI-compatible Chat Completions base. None → OpenAI default host.
    chat_api_base: str | None = None
    # True → native OpenAI Responses API. False → Chat Completions.
    uses_responses: bool = False
    cli_prefix: str = ""  # e.g. "groq/" for --worker-model groq/llama-...
    api_key_url: str = ""
    setup_instructions: str = ""


@dataclass(frozen=True)
class ModelInfo:
    """Per-model metadata."""

    model_id: str  # API model identifier
    provider_key: str  # Must match a ProviderInfo.provider_key
    default_tokens: int = 64000
    max_tokens: int = 128000
    uses_max_completion_tokens: bool = False  # GPT-5 series output cap
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
        uses_responses=True,
        cli_prefix="openai/",
        api_key_url="https://platform.openai.com/account/api-keys",
        setup_instructions=(
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) "
            "and chosen model below 🔑\n"
            "2. Provide details of the application that you would like to threat model  📝\n"
            "3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀"
        ),
    ),
    "Mistral": ProviderInfo(
        name="Mistral",
        provider_key="Mistral API",
        env_var="MISTRAL_API_KEY",
        chat_api_base="https://api.mistral.ai/v1",
        cli_prefix="mistral/",
        api_key_url="https://console.mistral.ai/api-keys/",
        setup_instructions=(
            "1. Enter your [Mistral API key](https://console.mistral.ai/api-keys/) "
            "and chosen model below 🔑\n"
            "2. Provide details of the application that you would like to threat model  📝\n"
            "3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀"
        ),
    ),
    "Groq": ProviderInfo(
        name="Groq",
        provider_key="Groq API",
        env_var="GROQ_API_KEY",
        chat_api_base="https://api.groq.com/openai/v1",
        cli_prefix="groq/",
        api_key_url="https://console.groq.com/keys",
        setup_instructions=(
            "1. Enter your [Groq API key](https://console.groq.com/keys) "
            "and chosen model below 🔑\n"
            "2. Provide details of the application that you would like to threat model  📝\n"
            "3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀"
        ),
    ),
    "DeepSeek": ProviderInfo(
        name="DeepSeek",
        provider_key="DeepSeek API",
        env_var="DEEPSEEK_API_KEY",
        chat_api_base="https://api.deepseek.com",
        cli_prefix="deepseek/",
        api_key_url="https://platform.deepseek.com/api_keys",
        setup_instructions=(
            "1. Enter your [DeepSeek API key](https://platform.deepseek.com/api_keys) "
            "and chosen model below 🔑\n"
            "2. Provide details of the application that you would like to threat model  📝\n"
            "3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀"
        ),
    ),
    "LM Studio": ProviderInfo(
        name="LM Studio",
        provider_key="LM Studio Server",
        env_var=None,
        needs_api_key=False,
        needs_api_base=True,
        default_api_base="http://localhost:1234",
        setup_instructions=(
            "1. Configure your LM Studio Server endpoint below (defaults to http://localhost:1234) 🔧\n"
            "2. Optionally enter an API key if your LM Studio Server requires authentication 🔑\n"
            "3. Provide details of the application that you would like to threat model 📝\n"
            "4. Generate a threat list, attack tree and/or mitigating controls for your application 🚀"
        ),
    ),
}


# ---------------------------------------------------------------------------
# Model registry — order determines UI display order per provider
# ---------------------------------------------------------------------------

MODELS: list[ModelInfo] = [
    # --- OpenAI ---
    ModelInfo(
        model_id="gpt-5.5",
        provider_key="OpenAI API",
        default_tokens=128000,
        max_tokens=1050000,
        uses_max_completion_tokens=True,
        help_text="GPT-5.5 is OpenAI's latest flagship model.",
    ),
    ModelInfo(
        model_id="gpt-5.4",
        provider_key="OpenAI API",
        default_tokens=128000,
        max_tokens=1050000,
        uses_max_completion_tokens=True,
        help_text="GPT-5.4 is OpenAI's previous flagship model with 1M+ context.",
    ),
    ModelInfo(
        model_id="gpt-5.4-pro",
        provider_key="OpenAI API",
        default_tokens=128000,
        max_tokens=1050000,
        uses_max_completion_tokens=True,
        help_text="GPT-5.4 Pro produces smarter, more precise responses.",
    ),
    ModelInfo(
        model_id="gpt-5.4-mini",
        provider_key="OpenAI API",
        default_tokens=64000,
        max_tokens=400000,
        uses_max_completion_tokens=True,
        help_text="GPT-5.4 Mini is a fast, cost-efficient version.",
    ),
    ModelInfo(
        model_id="gpt-5.4-nano",
        provider_key="OpenAI API",
        default_tokens=64000,
        max_tokens=400000,
        uses_max_completion_tokens=True,
        help_text="GPT-5.4 Nano is the fastest and most affordable option.",
    ),
    # --- Mistral ---
    ModelInfo(
        model_id="mistral-large-2512",
        provider_key="Mistral API",
        default_tokens=64000,
        max_tokens=128000,
        help_text="Mistral Large 3 offers premium capabilities.",
    ),
    ModelInfo(
        model_id="mistral-small-2603",
        provider_key="Mistral API",
        default_tokens=64000,
        max_tokens=256000,
        help_text="Mistral Small 4 merges reasoning, vision, and coding in a 256k-context model.",
    ),
    ModelInfo(
        model_id="mistral-medium-3-5",
        provider_key="Mistral API",
        default_tokens=64000,
        max_tokens=128000,
        help_text="Mistral Medium 3.5 provides balanced performance.",
    ),
    ModelInfo(
        model_id="mistral-medium-2508",
        provider_key="Mistral API",
        default_tokens=64000,
        max_tokens=128000,
        help_text="Mistral Medium 3.1 provides balanced performance.",
    ),
    ModelInfo(
        model_id="magistral-medium-2509",
        provider_key="Mistral API",
        default_tokens=32000,
        max_tokens=40000,
        help_text="Magistral Medium is a reasoning-focused model.",
    ),
    # --- Groq ---
    ModelInfo(
        model_id="openai/gpt-oss-120b",
        provider_key="Groq API",
        default_tokens=64000,
        max_tokens=131072,
        help_text="GPT-OSS 120B is an open-source reasoning model on Groq.",
    ),
    ModelInfo(
        model_id="openai/gpt-oss-20b",
        provider_key="Groq API",
        default_tokens=64000,
        max_tokens=131072,
        help_text="GPT-OSS 20B is a fast open-source model on Groq.",
    ),
    ModelInfo(
        model_id="llama-3.3-70b-versatile",
        provider_key="Groq API",
        default_tokens=64000,
        max_tokens=131072,
        help_text="Llama 3.3 70B excels at general-purpose tasks.",
    ),
    ModelInfo(
        model_id="qwen/qwen3-32b",
        provider_key="Groq API",
        default_tokens=64000,
        max_tokens=131072,
        help_text="Qwen3 32B delivers balanced performance.",
    ),
    # --- DeepSeek ---
    ModelInfo(
        model_id="deepseek-v4-pro",
        provider_key="DeepSeek API",
        default_tokens=64000,
        max_tokens=128000,
        help_text="DeepSeek V4 Pro is a large MoE model with strong reasoning, coding, and long-context performance.",
    ),
    ModelInfo(
        model_id="deepseek-v4-flash",
        provider_key="DeepSeek API",
        default_tokens=64000,
        max_tokens=128000,
        help_text="DeepSeek V4 Flash is the fast, cost-efficient tier.",
    ),
    # --- LM Studio (no static models — discovered at runtime) ---
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


def provider_uses_responses(provider_key: str) -> bool:
    """True if this provider speaks the OpenAI Responses API."""
    provider = _PROVIDER_BY_KEY.get(provider_key)
    return bool(provider and provider.uses_responses)


def model_uses_completion_tokens(model_id: str) -> bool:
    """True if this model uses max_completion_tokens / max_output_tokens."""
    return model_id in _COMPLETION_TOKEN_IDS

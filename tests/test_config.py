"""Tests for stride_gpt.config — model-pair conversion and API-key resolution."""

from __future__ import annotations

import json

import pytest

from stride_gpt.config import (
    config_to_model_pair,
    get_api_key,
    load_config,
    save_config,
)


@pytest.fixture
def single_tier_config():
    return {
        "worker_provider": "Groq",
        "worker_provider_key": "Groq API",
        "worker_model": "llama-3.3-70b-versatile",
        "worker_api_base": None,
        "worker_max_tokens": None,
    }


@pytest.fixture
def tiered_config():
    return {
        "worker_provider": "Groq",
        "worker_provider_key": "Groq API",
        "worker_model": "llama-3.3-70b-versatile",
        "worker_api_base": None,
        "worker_max_tokens": None,
        "architect_provider": "OpenAI",
        "architect_provider_key": "OpenAI API",
        "architect_model": "gpt-5.5",
        "architect_api_base": None,
        "architect_max_tokens": None,
    }


class TestConfigToModelPair:
    def test_returns_none_when_worker_missing(self):
        assert config_to_model_pair({}) is None

    def test_single_tier_yields_no_architect(self, single_tier_config, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        pair = config_to_model_pair(single_tier_config)
        assert pair is not None
        assert pair.architect is None
        assert pair.tiered is False
        assert pair.worker.model_name == "llama-3.3-70b-versatile"
        assert pair.worker.api_key == "groq-key"
        assert pair.for_architect() is pair.worker

    def test_tiered_yields_architect(self, tiered_config, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
        pair = config_to_model_pair(tiered_config)
        assert pair is not None
        assert pair.tiered
        assert pair.architect is not None
        assert pair.architect.model_name == "gpt-5.5"
        assert pair.architect.api_key == "oai-key"
        assert pair.worker.api_key == "groq-key"

    def test_null_architect_model_treated_as_single_tier(self, tiered_config, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        tiered_config["architect_model"] = None
        pair = config_to_model_pair(tiered_config)
        assert pair is not None
        assert pair.architect is None
        assert pair.tiered is False


class TestGetApiKey:
    def test_worker_reads_provider_env_var(self, single_tier_config, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "real-key")
        assert get_api_key(single_tier_config, tier="worker") == "real-key"

    def test_architect_reads_own_env_var(self, tiered_config, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
        assert get_api_key(tiered_config, tier="architect") == "oai-key"

    def test_architect_falls_back_to_worker_only_when_same_provider(self, monkeypatch):
        cfg = {
            "worker_provider": "Groq",
            "worker_provider_key": "Groq API",
            "architect_provider": "Groq",
            "architect_provider_key": "Groq API",
        }
        monkeypatch.setenv("GROQ_API_KEY", "shared-key")
        assert get_api_key(cfg, tier="architect") == "shared-key"

    def test_architect_does_not_reuse_key_across_providers(self, monkeypatch):
        cfg = {
            "worker_provider": "Groq",
            "worker_provider_key": "Groq API",
            "architect_provider": "OpenAI",
            "architect_provider_key": "OpenAI API",
        }
        monkeypatch.setenv("GROQ_API_KEY", "groq-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Architect must NOT receive the Groq key.
        assert get_api_key(cfg, tier="architect") == ""


class TestGetApiKeyLocalProvider:
    """LM Studio doesn't authenticate, so it must resolve to no key at all —
    never a cloud key picked up from the environment (issue #159)."""

    @staticmethod
    def _lm_studio_config():
        return {
            "worker_provider": "LM Studio",
            "worker_provider_key": "LM Studio Server",
            "worker_model": "google/gemma-4-e4b",
            "worker_api_base": "http://localhost:1234",
        }

    def test_returns_empty_when_no_keys_set(self, monkeypatch):
        for var in ("GROQ_API_KEY", "STRIDE_GPT_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        assert get_api_key(self._lm_studio_config(), tier="worker") == ""

    def test_does_not_leak_cloud_key_to_local_endpoint(self, monkeypatch):
        # api_base can point at any host, so the worker fallback chain must not
        # hand a real cloud key to it.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai")
        monkeypatch.setenv("GROQ_API_KEY", "sk-real-groq")
        monkeypatch.setenv("STRIDE_GPT_API_KEY", "sk-real-generic")
        assert get_api_key(self._lm_studio_config(), tier="worker") == ""


class TestGetApiKeyWorkerFallback:
    """The worker tier's last-resort scan of common env vars, used when the
    provider's own env var is unset (legacy behaviour)."""

    @staticmethod
    def _clear_all(monkeypatch):
        for var in (
            "GROQ_API_KEY",
            "STRIDE_GPT_API_KEY",
            "OPENAI_API_KEY",
            "MISTRAL_API_KEY",
            "DEEPSEEK_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_falls_back_to_generic_env_var_when_provider_var_unset(
        self, single_tier_config, monkeypatch
    ):
        # Groq's own GROQ_API_KEY is unset, so the worker falls back
        # to the generic STRIDE_GPT_API_KEY.
        self._clear_all(monkeypatch)
        monkeypatch.setenv("STRIDE_GPT_API_KEY", "generic-key")
        assert get_api_key(single_tier_config, tier="worker") == "generic-key"

    def test_stride_gpt_key_takes_precedence_in_fallback_chain(
        self, single_tier_config, monkeypatch
    ):
        # STRIDE_GPT_API_KEY is checked before the provider-branded vars.
        self._clear_all(monkeypatch)
        monkeypatch.setenv("STRIDE_GPT_API_KEY", "first")
        monkeypatch.setenv("OPENAI_API_KEY", "second")
        assert get_api_key(single_tier_config, tier="worker") == "first"

    def test_provider_var_wins_over_generic_fallback(
        self, single_tier_config, monkeypatch
    ):
        # When the provider's own env var IS set, the generic fallback is never
        # consulted — no risk of the wrong key winning.
        self._clear_all(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "provider-key")
        monkeypatch.setenv("STRIDE_GPT_API_KEY", "generic-key")
        assert get_api_key(single_tier_config, tier="worker") == "provider-key"

    def test_returns_empty_string_when_no_key_available(
        self, single_tier_config, monkeypatch
    ):
        self._clear_all(monkeypatch)
        assert get_api_key(single_tier_config, tier="worker") == ""


class TestLoadConfig:
    def test_returns_none_when_no_worker_model(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / ".stride-gpt"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(json.dumps({"some_other_field": 1}))
        monkeypatch.setattr("stride_gpt.config.CONFIG_FILE", cfg_dir / "config.json")
        assert load_config() is None

    def test_round_trips_two_tier(self, tmp_path, monkeypatch, tiered_config):
        cfg_dir = tmp_path / ".stride-gpt"
        cfg_dir.mkdir()
        cfg_path = cfg_dir / "config.json"
        monkeypatch.setattr("stride_gpt.config.CONFIG_DIR", cfg_dir)
        monkeypatch.setattr("stride_gpt.config.CONFIG_FILE", cfg_path)

        save_config(tiered_config)
        loaded = load_config()
        assert loaded == tiered_config

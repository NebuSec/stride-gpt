"""Tests for stride_gpt.agent.context — token counting and compression."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from stride_gpt.agent.context import (
    DEFAULT_LIMITS,
    KEEP_RECENT,
    ContextManager,
)
from stride_gpt.core.schemas import LLMConfig, LLMResponse


def _make_config(model: str = "gpt-5.4") -> LLMConfig:
    return LLMConfig(provider="OpenAI API", model_name=model, api_key="test")


@pytest.fixture
def ctx():
    return ContextManager(config=_make_config())


@pytest.fixture
def ctx_small():
    """Context manager with a tiny limit so compression triggers easily."""
    return ContextManager(config=_make_config(), context_window=100)


# ---------------------------------------------------------------------------
# _infer_limit
# ---------------------------------------------------------------------------


class TestResolveLimit:
    def test_registered_gpt_model(self):
        from stride_gpt.models import get_model

        ctx = ContextManager(config=_make_config("gpt-5.4"))
        expected = get_model("OpenAI API", "gpt-5.4").max_tokens
        assert ctx.context_window == expected

    def test_registered_mistral_model(self):
        from stride_gpt.models import get_model

        cfg = LLMConfig(provider="Mistral API", model_name="mistral-small-2603", api_key="test")
        ctx = ContextManager(config=cfg)
        expected = get_model("Mistral API", "mistral-small-2603").max_tokens
        assert ctx.context_window == expected

    def test_unregistered_claude_falls_back_to_name_inference(self):
        # Custom Claude-named local model not in the registry → keyword match wins
        ctx = ContextManager(config=_make_config("claude-sonnet-4-5-20250929"))
        assert ctx.context_window == DEFAULT_LIMITS["claude"]

    def test_unknown_model(self):
        ctx = ContextManager(config=_make_config("some-obscure-model"))
        assert ctx.context_window == DEFAULT_LIMITS["default"]

    def test_explicit_override(self):
        ctx = ContextManager(config=_make_config(), context_window=50_000)
        assert ctx.context_window == 50_000


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_four_chars_per_token(self, ctx):
        msgs = [{"role": "user", "content": "a" * 400}]
        result = ctx.count_tokens(msgs)
        assert result == 100  # 400 chars / 4


# ---------------------------------------------------------------------------
# needs_compression
# ---------------------------------------------------------------------------


class TestNeedsCompression:
    def test_under_threshold(self, ctx):
        assert ctx.needs_compression([{"role": "user", "content": "hi"}]) is False

    def test_over_threshold(self, ctx_small):
        msgs = [{"role": "user", "content": "a" * 500}]
        assert ctx_small.needs_compression(msgs) is True


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


class TestCompress:
    def test_noop_when_few_messages(self, ctx_small, llm_config):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result = ctx_small.compress(llm_config, msgs)
        assert result == msgs

    @patch("stride_gpt.agent.context.call_llm")
    def test_compresses_old_messages(self, mock_call_llm, ctx_small, llm_config):
        mock_call_llm.return_value = LLMResponse(
            content="Summary: found auth issues in auth.py",
            thinking=None,
            reasoning=None,
            model="test",
        )
        # Build enough messages to trigger compression
        msgs = [{"role": "system", "content": "system prompt"}]
        msgs.extend(
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(12)
        )

        result = ctx_small.compress(llm_config, msgs)

        # Should have: system + summary + KEEP_RECENT recent messages
        assert len(result) == 1 + 1 + KEEP_RECENT
        assert result[0]["role"] == "system"
        assert "Summary" in result[1]["content"]
        mock_call_llm.assert_called_once()

    @patch("stride_gpt.agent.context.call_llm")
    def test_preserves_system_messages(self, mock_call_llm, ctx_small, llm_config):
        mock_call_llm.return_value = LLMResponse(
            content="compressed", thinking=None, reasoning=None, model="test"
        )
        msgs = [{"role": "system", "content": "important system prompt"}]
        msgs.extend({"role": "user", "content": f"msg {i}"} for i in range(12))

        result = ctx_small.compress(llm_config, msgs)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "important system prompt"

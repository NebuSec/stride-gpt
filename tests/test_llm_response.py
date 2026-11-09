"""Tests for parsing a Responses API response in stride_gpt.core.llm."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from stride_gpt.core.llm import _call_responses, _parse_tool_arguments
from stride_gpt.core.schemas import LLMConfig


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(provider="OpenAI API", model_name="gpt-6-sol", api_key="fake")


def _reply(**fields):
    fields.setdefault("output", [])
    fields.setdefault("output_text", "hello")
    return SimpleNamespace(**fields)


def _call(config, response, tools=None):
    client = MagicMock()
    client.responses.create.return_value = response
    with patch("stride_gpt.core.llm.OpenAI", return_value=client):
        return _call_responses(config, [{"role": "user", "content": "hi"}], tools=tools)


class TestToolArgumentParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"path": "app.py"}', {"path": "app.py"}),
            ({"path": "app.py"}, {"path": "app.py"}),
            (None, {}),
            ("", {}),
            ("  \n", {}),
        ],
    )
    def test_accepts(self, raw, expected):
        assert _parse_tool_arguments(raw) == (expected, None)

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ('{"pattern": "a"b"}', "arguments were not valid JSON (Expecting ',' delimiter"),
            ('{"path": "app.', "arguments were not valid JSON (Unterminated string"),
            ('["app.py"]', "arguments must be a JSON object, not list"),
            ("42", "arguments must be a JSON object, not int"),
        ],
    )
    def test_rejects_without_raising(self, raw, message):
        arguments, error = _parse_tool_arguments(raw)
        assert arguments == {}
        assert error.startswith(message)

    def test_bad_call_does_not_fail_the_response(self, config):
        def tool_call(call_id, name, arguments):
            return SimpleNamespace(
                type="function_call", call_id=call_id, name=name, arguments=arguments
            )

        response = _call(
            config,
            _reply(
                output_text="",
                output=[
                    tool_call("a", "grep_content", '{"pattern": "x'),
                    tool_call("b", "read_file", '{"path": "app.py"}'),
                ],
            ),
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )

        bad, good = response.tool_calls
        assert (bad.id, bad.arguments) == ("a", {})
        assert bad.parse_error.startswith("arguments were not valid JSON")
        assert (good.arguments, good.parse_error) == ({"path": "app.py"}, None)


class TestUsageExtraction:
    """Reported usage lands on LLMResponse; none leaves it None, never 0 -- a
    caller must be able to tell "unknown" from "free"."""

    def test_records_usage_when_reported(self, config):
        usage = SimpleNamespace(input_tokens=120, output_tokens=30)
        response = _call(config, _reply(usage=usage))
        assert response.prompt_tokens == 120
        assert response.completion_tokens == 30

    def test_leaves_usage_none_when_absent(self, config):
        response = _call(config, _reply())
        assert response.prompt_tokens is None
        assert response.completion_tokens is None

    def test_zero_filled_usage_reads_as_unknown(self, config):
        usage = SimpleNamespace(input_tokens=0, output_tokens=0)
        response = _call(config, _reply(usage=usage))
        assert response.prompt_tokens is None
        assert response.completion_tokens is None

    def test_a_usage_object_missing_one_field_leaves_only_that_one_none(self, config):
        response = _call(config, _reply(usage=SimpleNamespace(input_tokens=42)))
        assert response.prompt_tokens == 42
        assert response.completion_tokens is None

    def test_no_cache_report_leaves_the_split_none(self, config):
        usage = SimpleNamespace(input_tokens=120, output_tokens=30)
        response = _call(config, _reply(usage=usage))
        assert response.cached_prompt_tokens is None
        assert response.uncached_prompt_tokens is None

    def test_cached_tokens_are_split_out(self, config):
        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=50,
            input_tokens_details=SimpleNamespace(cached_tokens=900),
        )
        response = _call(config, _reply(usage=usage))
        assert response.cached_prompt_tokens == 900
        assert response.uncached_prompt_tokens == 100

    def test_a_reported_zero_is_no_cache_hits_not_unknown(self, config):
        usage = SimpleNamespace(
            input_tokens=500,
            output_tokens=20,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        response = _call(config, _reply(usage=usage))
        assert response.cached_prompt_tokens == 0
        assert response.uncached_prompt_tokens == 500

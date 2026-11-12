"""Tests for the kwargs stride_gpt.core.llm hands to the OpenAI SDK."""

from __future__ import annotations

import pytest

from stride_gpt.core.llm import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    _client_kwargs,
    _responses_call_kwargs,
    _to_responses_tools,
    assistant_tool_turn,
    messages_to_responses_input,
)
from stride_gpt.core.schemas import LLMConfig, LLMResponse, ToolCallResult


class TestClientKwargs:
    def test_openai_uses_the_default_host(self):
        cfg = LLMConfig(provider="OpenAI API", model_name="gpt-6-sol", api_key="fake")
        kwargs = _client_kwargs(cfg)
        assert kwargs["api_key"] == "fake"
        assert kwargs["max_retries"] == 3
        assert "base_url" not in kwargs

    def test_custom_base_url_is_passed_through(self):
        cfg = LLMConfig(
            provider="OpenAI API",
            model_name="gpt-6-sol",
            api_key="fake",
            api_base="https://gateway.example/v1/",
        )
        assert _client_kwargs(cfg)["base_url"] == "https://gateway.example/v1"

    @pytest.mark.parametrize("provider", ["Groq API", "LM Studio Server", "Anthropic API"])
    def test_a_dropped_provider_from_a_saved_config_is_refused(self, provider):
        """It must not reach OpenAI's host with that provider's key and model."""
        cfg = LLMConfig(provider=provider, model_name="m", api_key="fake")
        with pytest.raises(ValueError, match="/config"):
            _client_kwargs(cfg)


class TestResponsesCallKwargs:
    def test_store_disabled_and_max_output_tokens(self):
        cfg = LLMConfig(provider="OpenAI API", model_name="gpt-6-sol", api_key="fake")
        kwargs = _responses_call_kwargs(cfg, [{"role": "user", "content": "hi"}])
        assert kwargs["store"] is False
        assert kwargs["model"] == "gpt-6-sol"
        assert kwargs["max_output_tokens"] == 128000
        assert "max_tokens" not in kwargs
        assert "max_completion_tokens" not in kwargs

    def test_json_mode_uses_text_format(self):
        cfg = LLMConfig(
            provider="OpenAI API",
            model_name="gpt-6-sol",
            api_key="fake",
            response_format="json",
        )
        kwargs = _responses_call_kwargs(cfg, [{"role": "user", "content": "hi"}])
        assert kwargs["text"] == {"format": {"type": "json_object"}}

    def test_tools_skip_json_format(self):
        cfg = LLMConfig(
            provider="OpenAI API",
            model_name="gpt-6-sol",
            api_key="fake",
            response_format="json",
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        kwargs = _responses_call_kwargs(
            cfg, [{"role": "user", "content": "hi"}], tools=tools
        )
        assert "text" not in kwargs
        assert kwargs["tools"] == [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            }
        ]


class TestMessagesToResponsesInput:
    def test_system_becomes_instructions(self):
        instructions, items = messages_to_responses_input(
            [
                {"role": "system", "content": "You are a threat modeller."},
                {"role": "user", "content": "analyse auth.py"},
            ]
        )
        assert instructions == "You are a threat modeller."
        assert items == [{"role": "user", "content": "analyse auth.py"}]

    def test_tool_results_become_function_call_output(self):
        instructions, items = messages_to_responses_input(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "auth.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "read_file",
                    "content": "def login(): pass",
                },
            ]
        )
        assert instructions is None
        assert items[0]["type"] == "function_call"
        assert items[0]["call_id"] == "call_1"
        assert items[0]["name"] == "read_file"
        assert items[1] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "def login(): pass",
        }

    def test_image_parts_converted(self):
        _, items = messages_to_responses_input(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ],
                }
            ]
        )
        assert items[0]["content"] == [
            {"type": "input_text", "text": "describe"},
            {"type": "input_image", "image_url": "data:image/png;base64,abc"},
        ]


class TestToResponsesTools:
    def test_flattens_nested_chat_completions_shape(self):
        nested = [
            {
                "type": "function",
                "function": {
                    "name": "grep_content",
                    "description": "Search files",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            }
        ]
        assert _to_responses_tools(nested) == [
            {
                "type": "function",
                "name": "grep_content",
                "description": "Search files",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            }
        ]


class TestCallTimeout:
    def test_defaults_when_unset(self):
        config = LLMConfig(provider="OpenAI API", model_name="gpt-6-sol", api_key="k")
        assert _client_kwargs(config)["timeout"] == DEFAULT_CALL_TIMEOUT_SECONDS

    def test_config_sets_it(self):
        config = LLMConfig(provider="OpenAI API", model_name="gpt-6-sol", api_key="k", timeout=90)
        assert _client_kwargs(config)["timeout"] == 90


_ITEM = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"}


class TestReasoningReplay:

    def _response(self, reasoning):
        return LLMResponse(
            content="",
            tool_calls=[ToolCallResult(id="call_1", function_name="read_file",
                                       arguments={"path": "a.py"})],
            reasoning_items=reasoning,
        )

    def test_tool_turn_carries_the_reasoning(self):
        turn = assistant_tool_turn(self._response([_ITEM]))
        assert turn["reasoning_items"] == [_ITEM]
        assert turn["tool_calls"][0]["function"] == {
            "name": "read_file", "arguments": '{"path": "a.py"}'}

    def test_tool_turn_without_reasoning_has_no_key(self):
        assert "reasoning_items" not in assistant_tool_turn(self._response(None))

    def test_reasoning_goes_back_before_its_calls(self):
        _, items = messages_to_responses_input([
            {"role": "user", "content": "go"},
            assistant_tool_turn(self._response([_ITEM])),
            {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "x"},
        ])
        assert [item.get("type", item.get("role")) for item in items] == [
            "user", "reasoning", "function_call", "function_call_output"]
        assert items[1] == _ITEM

    def test_reasoning_models_ask_for_encrypted_reasoning(self):
        config = LLMConfig(provider="OpenAI API", model_name="gpt-6-sol", api_key="k")
        kwargs = _responses_call_kwargs(config, [{"role": "user", "content": "hi"}])
        assert kwargs["include"] == ["reasoning.encrypted_content"]

    def test_an_unknown_route_asks_too(self):
        config = LLMConfig(provider="OpenAI API", model_name="some-gateway-route", api_key="k")
        kwargs = _responses_call_kwargs(config, [{"role": "user", "content": "hi"}])
        assert kwargs["include"] == ["reasoning.encrypted_content"]

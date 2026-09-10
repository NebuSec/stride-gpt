"""Regression tests for OpenAI SDK client / call kwargs."""

from __future__ import annotations

from stride_gpt.core.llm import (
    _chat_call_kwargs,
    _client_kwargs,
    _responses_call_kwargs,
    _to_responses_tools,
    messages_to_responses_input,
)
from stride_gpt.core.schemas import LLMConfig


class TestClientKwargs:
    def test_lm_studio_placeholder_api_key_and_v1_base(self):
        """LM Studio: OpenAI SDK refuses to build a client without an api_key
        even for localhost. A missing key must become a placeholder
        (issue #159), and /v1 is appended to the user-supplied host."""
        cfg = LLMConfig(
            provider="LM Studio Server",
            model_name="google/gemma-4-e4b",
            api_key="",
            api_base="http://localhost:1234",
        )
        kwargs = _client_kwargs(cfg)
        assert kwargs["api_key"] == "not-needed"
        assert kwargs["base_url"] == "http://localhost:1234/v1"

    def test_lm_studio_real_key_preserved(self):
        cfg = LLMConfig(
            provider="LM Studio Server",
            model_name="google/gemma-4-e4b",
            api_key="sk-user-supplied",
            api_base="http://localhost:1234",
        )
        kwargs = _client_kwargs(cfg)
        assert kwargs["api_key"] == "sk-user-supplied"

    def test_cloud_provider_without_key_still_passes_none(self):
        """Placeholder is scoped to non-authenticating providers."""
        cfg = LLMConfig(provider="OpenAI API", model_name="gpt-5.5", api_key="")
        kwargs = _client_kwargs(cfg)
        assert kwargs["api_key"] is None
        assert "base_url" not in kwargs

    def test_deepseek_uses_hosted_base_url(self):
        cfg = LLMConfig(
            provider="DeepSeek API",
            model_name="deepseek-v4-pro",
            api_key="fake",
        )
        kwargs = _client_kwargs(cfg)
        assert kwargs["base_url"] == "https://api.deepseek.com"

    def test_groq_uses_hosted_base_url(self):
        cfg = LLMConfig(
            provider="Groq API",
            model_name="llama-3.3-70b-versatile",
            api_key="fake",
        )
        kwargs = _client_kwargs(cfg)
        assert kwargs["base_url"] == "https://api.groq.com/openai/v1"


class TestChatCallKwargs:
    def test_deepseek_leaves_max_tokens_to_provider(self):
        cfg = LLMConfig(
            provider="DeepSeek API",
            model_name="deepseek-v4-flash",
            api_key="fake",
        )
        kwargs = _chat_call_kwargs(cfg, [{"role": "user", "content": "hi"}])
        assert kwargs["model"] == "deepseek-v4-flash"
        assert "max_tokens" not in kwargs
        assert "max_completion_tokens" not in kwargs

    def test_lm_studio_omits_json_object(self):
        cfg = LLMConfig(
            provider="LM Studio Server",
            model_name="local-model",
            api_key="not-needed",
            api_base="http://localhost:1234",
            response_format="json",
        )
        kwargs = _chat_call_kwargs(cfg, [{"role": "user", "content": "hi"}])
        assert "response_format" not in kwargs

    def test_json_object_sent_for_compatible_cloud(self):
        cfg = LLMConfig(
            provider="Mistral API",
            model_name="mistral-small-2603",
            api_key="fake",
            response_format="json",
        )
        kwargs = _chat_call_kwargs(cfg, [{"role": "user", "content": "hi"}])
        assert kwargs["response_format"] == {"type": "json_object"}


class TestResponsesCallKwargs:
    def test_store_disabled_and_max_output_tokens(self):
        cfg = LLMConfig(provider="OpenAI API", model_name="gpt-5.5", api_key="fake")
        kwargs = _responses_call_kwargs(cfg, [{"role": "user", "content": "hi"}])
        assert kwargs["store"] is False
        assert kwargs["model"] == "gpt-5.5"
        assert kwargs["max_output_tokens"] == 128000
        assert "max_tokens" not in kwargs
        assert "max_completion_tokens" not in kwargs

    def test_json_mode_uses_text_format(self):
        cfg = LLMConfig(
            provider="OpenAI API",
            model_name="gpt-5.5",
            api_key="fake",
            response_format="json",
        )
        kwargs = _responses_call_kwargs(cfg, [{"role": "user", "content": "hi"}])
        assert kwargs["text"] == {"format": {"type": "json_object"}}

    def test_tools_skip_json_format(self):
        cfg = LLMConfig(
            provider="OpenAI API",
            model_name="gpt-5.5",
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

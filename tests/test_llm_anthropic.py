"""The Anthropic Messages side of stride_gpt.core.llm (core/llm_anthropic.py).

The streaming tests run the real Anthropic client against a mock transport, so
they check what actually goes over the wire.
"""

from __future__ import annotations

import itertools
import json

import anthropic
import httpx
import httpx2
import openai
import pytest

from stride_gpt.agent.errors import classify_error
from stride_gpt.config import get_api_key
from stride_gpt.core import llm, llm_anthropic
from stride_gpt.core.llm import (
    ModelRefusalError,
    assistant_tool_turn,
    call_llm,
    call_llm_with_tools,
    messages_to_responses_input,
)
from stride_gpt.core.llm_anthropic import (
    DEFAULT_MAX_TOKENS,
    messages_call_kwargs,
    messages_to_anthropic,
    request_binding,
)
from stride_gpt.core.schemas import LLMConfig

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]
_OTHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _config(model: str = "claude-opus-5-5", **kw) -> LLMConfig:
    return LLMConfig(provider="Anthropic API", model_name=model, api_key="sk-ant-test", **kw)


# ---------------------------------------------------------------------------
# A scripted Messages endpoint
# ---------------------------------------------------------------------------


def _sse(*events: dict) -> bytes:
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events
    )


def _message_events(
    *blocks: dict,
    stop_reason: str = "end_turn",
    usage: dict | None = None,
    stop_details: dict | None = None,
) -> list[dict]:
    """The SSE events of one Messages response with ``blocks`` as its content."""
    start_usage = {"input_tokens": 10, "output_tokens": 1, **(usage or {})}
    events: list[dict] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": start_usage,
            },
        }
    ]
    for index, block in enumerate(blocks):
        kind = block["type"]
        if kind == "text":
            events += [
                {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block["text"]}},
            ]
        elif kind == "thinking":
            events += [
                {"type": "content_block_start", "index": index,
                 "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                {"type": "content_block_delta", "index": index,
                 "delta": {"type": "thinking_delta", "thinking": block["thinking"]}},
                {"type": "content_block_delta", "index": index,
                 "delta": {"type": "signature_delta", "signature": block["signature"]}},
            ]
        elif kind == "tool_use":
            events += [
                {"type": "content_block_start", "index": index,
                 "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}},
                {"type": "content_block_delta", "index": index,
                 "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}},
            ]
        events.append({"type": "content_block_stop", "index": index})
    delta: dict = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    events += [
        {"type": "message_delta", "delta": delta, "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    ]
    return events


_OK = _message_events({"type": "text", "text": "ok"})
_OVERLOADED = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}


@pytest.fixture
def no_sleep(monkeypatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(llm, "_sleep", slept.append)
    monkeypatch.setattr("anthropic._base_client.time.sleep", slept.append)
    return slept


def _serve(monkeypatch, outcomes: list) -> list[httpx2.Request]:
    """Serve each request the next outcome: a list of SSE events (a 200
    stream), or (status, body) for an error response."""
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        outcome = outcomes.pop(0)
        if isinstance(outcome, tuple):
            status, body = outcome
            return httpx2.Response(status, json=body)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse(*outcome)
        )

    real = anthropic.Anthropic

    def client(**kwargs):
        return real(http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(llm_anthropic, "Anthropic", client)
    return calls


def _body(request: httpx2.Request) -> dict:
    return json.loads(request.content)


# ---------------------------------------------------------------------------
# Request kwargs
# ---------------------------------------------------------------------------


class TestCallKwargs:
    def test_registered_model_defaults(self):
        kwargs = messages_call_kwargs(
            _config(), [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        )
        assert kwargs["model"] == "claude-opus-5-5"
        assert kwargs["max_tokens"] == 64000
        assert kwargs["system"] == "sys"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["cache_control"] == {"type": "ephemeral"}
        assert "thinking" not in kwargs
        assert "output_config" not in kwargs

    def test_an_unregistered_model_still_gets_the_required_cap(self):
        kwargs = messages_call_kwargs(_config("gateway-route"), [{"role": "user", "content": "hi"}])
        assert kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
        assert "fallbacks" not in kwargs

    def test_an_explicit_cap_applies(self):
        kwargs = messages_call_kwargs(_config(max_tokens=1234), [{"role": "user", "content": "hi"}])
        assert kwargs["max_tokens"] == 1234

    def test_effort_goes_in_output_config(self):
        kwargs = messages_call_kwargs(
            _config(reasoning_effort="xhigh"), [{"role": "user", "content": "hi"}]
        )
        assert kwargs["output_config"] == {"effort": "xhigh"}

    @pytest.mark.parametrize("effort", ["none", "minimal"])
    def test_a_responses_only_effort_is_refused(self, effort):
        with pytest.raises(ValueError, match="Messages API"):
            messages_call_kwargs(_config(reasoning_effort=effort), [{"role": "user", "content": "hi"}])

    def test_a_schema_becomes_strict_structured_output(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        kwargs = messages_call_kwargs(
            _config(response_format=schema), [{"role": "user", "content": "hi"}]
        )
        fmt = kwargs["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"]["additionalProperties"] is False

    def test_json_mode_sends_nothing(self):
        kwargs = messages_call_kwargs(
            _config(response_format="json"), [{"role": "user", "content": "hi"}]
        )
        assert "output_config" not in kwargs

    def test_tools_skip_the_output_format(self):
        kwargs = messages_call_kwargs(
            _config(response_format={"type": "object", "properties": {}}),
            [{"role": "user", "content": "hi"}],
            _TOOLS,
        )
        assert "output_config" not in kwargs
        assert kwargs["tools"] == [
            {
                "name": "read_file",
                "description": "Read a file.",
                "input_schema": _TOOLS[0]["function"]["parameters"],
            }
        ]

    def test_refusal_fallbacks_follow_the_registry(self):
        opus = messages_call_kwargs(_config(), [{"role": "user", "content": "hi"}])
        haiku = messages_call_kwargs(_config("claude-haiku-5-5"), [{"role": "user", "content": "hi"}])
        assert opus["fallbacks"] == "default"
        assert opus["betas"] == ["server-side-fallback-2026-07-01"]
        assert "fallbacks" not in haiku and "betas" not in haiku

    def test_no_refusal_fallbacks_through_a_gateway(self):
        kwargs = messages_call_kwargs(
            _config(api_base="https://gateway.example"), [{"role": "user", "content": "hi"}]
        )
        assert "fallbacks" not in kwargs and "betas" not in kwargs


# ---------------------------------------------------------------------------
# History conversion
# ---------------------------------------------------------------------------


def _tool_turn(call_id: str = "toolu_1", reasoning_items=None) -> dict:
    msg = {
        "role": "assistant",
        "content": "Looking.",
        "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
        ],
    }
    if reasoning_items is not None:
        msg["reasoning_items"] = reasoning_items
    return msg


class TestMessagesToAnthropic:
    def test_parallel_tool_results_share_one_user_message(self):
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                    {"id": "t2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "one"},
            {"role": "tool", "tool_call_id": "t2", "content": "two"},
        ]
        _, out = messages_to_anthropic(messages, model="m")
        assert [m["role"] for m in out] == ["user", "assistant", "user"]
        assert [b["type"] for b in out[1]["content"]] == ["tool_use", "tool_use"]
        assert out[2]["content"] == [
            {"type": "tool_result", "tool_use_id": "t1", "content": "one"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "two"},
        ]

    def test_tool_call_arguments_become_an_input_object(self):
        _, out = messages_to_anthropic([{"role": "user", "content": "go"}, _tool_turn()], model="m")
        assert out[1]["content"] == [
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}},
        ]

    def test_a_data_url_image_is_sent_as_base64(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ]
        _, out = messages_to_anthropic(messages, model="m")
        assert out[0]["content"][1] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
        }

    def test_a_system_only_request_is_sent_as_a_user_message(self):
        system, out = messages_to_anthropic([{"role": "system", "content": "do it"}], model="m")
        assert system is None
        assert out == [{"role": "user", "content": "do it"}]

    def test_a_later_system_message_stays_in_place(self):
        system, out = messages_to_anthropic(
            [
                {"role": "system", "content": "s1"},
                {"role": "user", "content": "u"},
                {"role": "system", "content": "s2"},
            ],
            model="m",
        )
        assert system == "s1"
        assert out[-1] == {"role": "system", "content": "s2"}


# ---------------------------------------------------------------------------
# Thinking replay and its binding to the request
# ---------------------------------------------------------------------------

_THINKING = {"type": "thinking", "thinking": "I should read a.py", "signature": "sig-1"}
_TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}}


class TestThinkingReplay:
    def _first_round(self, monkeypatch, no_sleep, *, system="sys", tools=_TOOLS):
        _serve(monkeypatch, [_message_events(_THINKING, _TOOL_USE, stop_reason="tool_use")])
        history = [{"role": "system", "content": system}, {"role": "user", "content": "go"}]
        response = call_llm_with_tools(_config(), history, tools)
        history.append(assistant_tool_turn(response))
        history.append({"role": "tool", "tool_call_id": "toolu_1", "content": "print(1)"})
        return history

    def test_an_unchanged_conversation_replays_the_turn_with_its_thinking(self, monkeypatch, no_sleep):
        history = self._first_round(monkeypatch, no_sleep)
        kwargs = messages_call_kwargs(_config(), history, _TOOLS)
        assert kwargs["messages"][1] == {"role": "assistant", "content": [_THINKING, _TOOL_USE]}

    def test_changed_tools_drop_the_thinking(self, monkeypatch, no_sleep):
        """The grace round swaps the tool set; the API would refuse the block."""
        history = self._first_round(monkeypatch, no_sleep)
        kwargs = messages_call_kwargs(_config(), history, _OTHER_TOOLS)
        assert kwargs["messages"][1]["content"] == [_TOOL_USE]

    def test_an_edited_earlier_message_drops_the_thinking(self, monkeypatch, no_sleep):
        """Context compression rewrites the task message into a summary."""
        history = self._first_round(monkeypatch, no_sleep)
        history[1] = {"role": "user", "content": "go\n\n[summary]"}
        kwargs = messages_call_kwargs(_config(), history, _TOOLS)
        assert kwargs["messages"][1]["content"] == [_TOOL_USE]

    def test_another_model_drops_the_thinking(self, monkeypatch, no_sleep):
        history = self._first_round(monkeypatch, no_sleep)
        kwargs = messages_call_kwargs(_config("claude-sonnet-5-5"), history, _TOOLS)
        assert kwargs["messages"][1]["content"] == [_TOOL_USE]

    def test_the_binding_hashes_what_was_sent(self, monkeypatch, no_sleep):
        calls = _serve(monkeypatch, [_message_events(_THINKING, _TOOL_USE, stop_reason="tool_use")])
        history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
        response = call_llm_with_tools(_config(), history, _TOOLS)
        sent = _body(calls[0])
        assert response.reasoning_items[0]["binding"] == request_binding(sent)

    def test_the_responses_path_skips_claude_turns(self, monkeypatch, no_sleep):
        history = self._first_round(monkeypatch, no_sleep)
        _, items = messages_to_responses_input(history)
        assert not any(item.get("type") == "anthropic_turn" for item in items)


# ---------------------------------------------------------------------------
# Streaming over the wire
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_call_llm_routes_to_the_messages_api(self, monkeypatch, no_sleep):
        calls = _serve(monkeypatch, [_OK])
        response = call_llm(_config("claude-haiku-5-5"), [{"role": "user", "content": "hi"}])
        assert response.content == "ok"
        assert calls[0].url.path == "/v1/messages"
        body = _body(calls[0])
        assert body["stream"] is True
        assert calls[0].headers["authorization"] == "Bearer sk-ant-test"

    def test_fallback_models_send_the_beta(self, monkeypatch, no_sleep):
        calls = _serve(monkeypatch, [_OK])
        call_llm(_config(), [{"role": "user", "content": "hi"}])
        assert "server-side-fallback-2026-07-01" in calls[0].headers["anthropic-beta"]
        assert _body(calls[0])["fallbacks"] == "default"

    def test_a_custom_base_url_is_used(self, monkeypatch, no_sleep):
        calls = _serve(monkeypatch, [_OK])
        call_llm(
            _config("claude-haiku-5-5", api_base="https://gateway.example/anthropic/"),
            [{"role": "user", "content": "hi"}],
        )
        assert str(calls[0].url) == "https://gateway.example/anthropic/v1/messages"

    def test_text_tools_thinking_and_usage_are_read(self, monkeypatch, no_sleep):
        _serve(
            monkeypatch,
            [
                _message_events(
                    _THINKING,
                    {"type": "text", "text": "Reading."},
                    _TOOL_USE,
                    stop_reason="tool_use",
                    usage={"cache_read_input_tokens": 100, "cache_creation_input_tokens": 5},
                )
            ],
        )
        response = call_llm_with_tools(_config(), [{"role": "user", "content": "go"}], _TOOLS)
        assert response.content == "Reading."
        assert response.thinking == "I should read a.py"
        assert [(tc.id, tc.function_name, tc.arguments) for tc in response.tool_calls] == [
            ("toolu_1", "read_file", {"path": "a.py"})
        ]
        assert response.prompt_tokens == 115
        assert response.cached_prompt_tokens == 100
        assert response.uncached_prompt_tokens == 15
        assert response.completion_tokens == 20
        assert response.incomplete_reason is None

    def test_a_max_tokens_stop_is_marked_incomplete(self, monkeypatch, no_sleep, caplog):
        _serve(monkeypatch, [_message_events({"type": "text", "text": "par"}, stop_reason="max_tokens")])
        response = call_llm(_config(), [{"role": "user", "content": "hi"}])
        assert response.incomplete_reason == "max_output_tokens"
        assert "incomplete" in caplog.text

    def test_a_refusal_is_raised(self, monkeypatch, no_sleep):
        _serve(
            monkeypatch,
            [
                _message_events(
                    stop_reason="refusal",
                    stop_details={"type": "refusal", "category": "cyber", "explanation": "No."},
                )
            ],
        )
        with pytest.raises(ModelRefusalError, match=r"No\."):
            call_llm(_config(), [{"role": "user", "content": "hi"}])

    def test_an_overloaded_stream_is_retried(self, monkeypatch, no_sleep):
        dead = [*_message_events({"type": "text", "text": "o"})[:3], _OVERLOADED]
        calls = _serve(monkeypatch, [dead, _OK])
        assert call_llm(_config(), [{"role": "user", "content": "hi"}]).content == "ok"
        assert len(calls) == 2

    def test_a_stream_that_ends_early_is_retried(self, monkeypatch, no_sleep):
        cut = _message_events({"type": "text", "text": "o"})[:3]
        calls = _serve(monkeypatch, [cut, _OK])
        assert call_llm(_config(), [{"role": "user", "content": "hi"}]).content == "ok"
        assert len(calls) == 2

    def test_gives_up_after_three_dead_streams(self, monkeypatch, no_sleep):
        dead = [*_OK[:1], _OVERLOADED]
        calls = _serve(monkeypatch, [dead, dead, dead])
        with pytest.raises(anthropic.APIStatusError):
            call_llm(_config(), [{"role": "user", "content": "hi"}])
        assert len(calls) == llm.STREAM_ATTEMPTS

    def test_a_bad_request_is_not_retried(self, monkeypatch, no_sleep):
        error = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
        calls = _serve(monkeypatch, [(400, error)])
        with pytest.raises(anthropic.BadRequestError):
            call_llm(_config(), [{"role": "user", "content": "hi"}])
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Errors and keys
# ---------------------------------------------------------------------------


class TestErrors:
    def _status_error(self, monkeypatch, no_sleep, status: int, error_type: str, message: str):
        body = {"type": "error", "error": {"type": error_type, "message": message}}
        _serve(monkeypatch, [(status, body)] * (llm.SDK_RETRIES + 1))
        with pytest.raises(anthropic.APIError) as caught:
            call_llm(_config(), [{"role": "user", "content": "hi"}])
        return caught.value

    def test_prompt_too_long_is_a_context_overflow(self, monkeypatch, no_sleep):
        exc = self._status_error(
            monkeypatch, no_sleep, 400, "invalid_request_error", "prompt is too long: 1200000 tokens"
        )
        error_class, reason = classify_error(exc)
        assert error_class == "context_overflow"
        assert "prompt is too long" in reason

    def test_rate_limit_and_auth(self, monkeypatch, no_sleep):
        assert classify_error(
            self._status_error(monkeypatch, no_sleep, 429, "rate_limit_error", "slow down")
        )[0] == "rate_limited"
        assert classify_error(
            self._status_error(monkeypatch, no_sleep, 401, "authentication_error", "bad key")
        )[0] == "auth"

    def test_a_mid_stream_error_is_classified_by_its_type(self, monkeypatch, no_sleep):
        dead = [*_OK[:1], _OVERLOADED]
        _serve(monkeypatch, [dead] * llm.STREAM_ATTEMPTS)
        with pytest.raises(anthropic.APIError) as caught:
            call_llm(_config(), [{"role": "user", "content": "hi"}])
        assert classify_error(caught.value)[0] == "provider_error"


class TestKeys:
    def test_an_openai_key_is_never_offered_to_anthropic(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
        config = {"worker_provider": "Anthropic", "worker_provider_key": "Anthropic API"}
        assert get_api_key(config, tier="worker") == ""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        assert get_api_key(config, tier="worker") == "ant-key"

    def test_the_cli_prefix_resolves_the_provider(self):
        from stride_gpt.cli import _resolve_provider

        assert _resolve_provider("anthropic/claude-opus-5-5") == ("Anthropic API", "claude-opus-5-5")


def test_openai_sdk_errors_still_classify():
    """The Responses path is untouched by the new classes."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    exc = openai.APIConnectionError(request=request)
    assert classify_error(exc)[0] == "provider_error"


# ---------------------------------------------------------------------------
# Review fixes: transport faults, tool JSON, truncation, fallbacks, usage
# ---------------------------------------------------------------------------


def _serve_raw(monkeypatch, outcomes: list) -> list[httpx2.Request]:
    """Like ``_serve``, but an outcome may be (events, exception): the stream
    sends the events and then fails with the exception mid-body."""
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        outcome = outcomes.pop(0)
        if isinstance(outcome, tuple):
            events, error = outcome

            def body():
                yield _sse(*events)
                raise error

            content = body()
        else:
            content = _sse(*outcome)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    real = anthropic.Anthropic

    def client(**kwargs):
        return real(http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(llm_anthropic, "Anthropic", client)
    return calls


class TestReviewFixes:
    @pytest.mark.parametrize(
        "error", [httpx2.ReadError("reset"), httpx2.RemoteProtocolError("cut"), httpx2.ReadTimeout("slow")]
    )
    def test_a_transport_fault_mid_stream_is_retried(self, monkeypatch, no_sleep, error):
        calls = _serve_raw(monkeypatch, [(_OK[:3], error), _OK])
        assert call_llm(_config(), [{"role": "user", "content": "hi"}]).content == "ok"
        assert len(calls) == 2

    def test_a_transport_fault_that_persists_is_a_provider_error(self, monkeypatch, no_sleep):
        _serve_raw(monkeypatch, [(_OK[:3], httpx2.ReadError("reset"))] * llm.STREAM_ATTEMPTS)
        with pytest.raises(anthropic.APIConnectionError) as caught:
            call_llm(_config(), [{"role": "user", "content": "hi"}])
        assert classify_error(caught.value)[0] == "provider_error"

    def test_unparseable_tool_json_is_sampled_again(self, monkeypatch, no_sleep):
        bad = [
            *_message_events(stop_reason="tool_use")[:1],
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "input_json_delta", "partial_json": '{"path": }}'}},
            # The stream itself completes: only the JSON is at fault.
            *_message_events(stop_reason="tool_use")[-2:],
        ]
        calls = _serve(monkeypatch, [bad, _message_events(_TOOL_USE, stop_reason="tool_use")])
        response = call_llm_with_tools(_config(), [{"role": "user", "content": "go"}], _TOOLS)
        assert response.tool_calls[0].arguments == {"path": "a.py"}
        assert len(calls) == 2

    def test_a_tool_call_cut_off_by_the_output_cap_is_not_run(self, monkeypatch, no_sleep):
        _serve(monkeypatch, [_message_events(_TOOL_USE, stop_reason="max_tokens")])
        response = call_llm_with_tools(_config(), [{"role": "user", "content": "go"}], _TOOLS)
        (call,) = response.tool_calls
        assert call.arguments == {}
        assert "output token limit" in call.parse_error

    def test_an_empty_text_block_is_not_replayed(self, monkeypatch, no_sleep):
        _serve(monkeypatch, [_message_events({"type": "text", "text": ""}, _TOOL_USE, stop_reason="tool_use")])
        response = call_llm_with_tools(_config(), [{"role": "user", "content": "go"}], _TOOLS)
        assert response.reasoning_items[0]["content"] == [_TOOL_USE]

    def test_a_fallback_served_turn_is_replayed_without_its_thinking(self, monkeypatch, no_sleep):
        events = _message_events(_THINKING, _TOOL_USE, stop_reason="tool_use")
        events[-2]["usage"]["iterations"] = [
            {"type": "message", "input_tokens": 1, "output_tokens": 1},
            {"type": "fallback_message", "input_tokens": 1, "output_tokens": 1},
        ]
        _serve(monkeypatch, [events])
        history = [{"role": "user", "content": "go"}]
        response = call_llm_with_tools(_config(), history, _TOOLS)
        history.append(assistant_tool_turn(response))
        kwargs = messages_call_kwargs(_config(), history, _TOOLS)
        assert kwargs["messages"][1]["content"] == [_TOOL_USE]

    def test_anthropic_base_url_counts_as_a_gateway(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")
        kwargs = messages_call_kwargs(_config(), [{"role": "user", "content": "hi"}])
        assert "fallbacks" not in kwargs

    def test_unreported_cache_fields_leave_the_split_unknown(self, monkeypatch, no_sleep):
        _serve(monkeypatch, [_OK])
        response = call_llm(_config(), [{"role": "user", "content": "hi"}])
        assert response.prompt_tokens == 10
        assert response.cached_prompt_tokens is None
        assert response.uncached_prompt_tokens is None

    @pytest.mark.parametrize(
        ("error_type", "expected"),
        [("rate_limit_error", "rate_limited"), ("authentication_error", "auth"),
         ("permission_error", "auth"), ("overloaded_error", "provider_error")],
    )
    def test_a_mid_stream_error_type_decides_the_class(self, monkeypatch, no_sleep, error_type, expected):
        event = {"type": "error", "error": {"type": error_type, "message": "x"}}
        _serve(monkeypatch, [[*_OK[:1], event]] * llm.STREAM_ATTEMPTS)
        with pytest.raises(anthropic.APIError) as caught:
            call_llm(_config(), [{"role": "user", "content": "hi"}])
        assert classify_error(caught.value)[0] == expected

    def test_a_missing_sdk_credential_is_an_auth_failure(self):
        assert classify_error(anthropic.CredentialsError("no credentials"))[0] == "auth"


class TestCliModelPair:
    def test_a_responses_only_effort_is_refused_before_any_call(self, monkeypatch):
        import typer

        from stride_gpt import cli

        monkeypatch.setattr(cli, "load_config", lambda: None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        with pytest.raises(typer.Exit) as exc:
            cli._build_model_pair(
                worker_model="anthropic/claude-opus-5-5",
                worker_reasoning_effort="minimal",
                architect_model="openai/gpt-6-sol",
            )
        assert exc.value.exit_code == 2

    def test_a_saved_endpoint_of_another_provider_is_not_reused(self, monkeypatch):
        from stride_gpt import cli

        saved = {
            "worker_provider": "OpenAI",
            "worker_provider_key": "OpenAI API",
            "worker_model": "gpt-6-sol",
            "worker_api_base": "https://openai-proxy.internal/v1",
            "worker_max_tokens": 128000,
        }
        monkeypatch.setattr(cli, "load_config", lambda: saved)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        pair = cli._build_model_pair(worker_model="anthropic/claude-opus-5-5", no_architect=True)
        assert pair.worker.api_base is None
        assert pair.worker.max_tokens is None


# ---------------------------------------------------------------------------
# Second review: status codes, gateway bodies, unreplayable calls
# ---------------------------------------------------------------------------


class TestSecondReview:
    def test_a_gateway_billing_stop_keeps_its_status_and_wording(self, monkeypatch, no_sleep):
        """vega's runner pauses a scan on a 403 with the proxy's cost-cap wording."""
        body = {"detail": "insufficient balance: scan cost cap exceeded (spent $5.00 of $5.00 cap)"}
        _serve(monkeypatch, [(403, body)])
        with pytest.raises(anthropic.PermissionDeniedError) as caught:
            call_llm(_config(), [{"role": "user", "content": "hi"}])
        error_class, reason = classify_error(caught.value)
        assert error_class == "auth"
        assert "(HTTP 403: insufficient balance: scan cost cap exceeded" in reason

    def test_a_streamed_failure_has_no_status_to_add(self, monkeypatch, no_sleep):
        _serve(monkeypatch, [[*_OK[:1], _OVERLOADED]] * llm.STREAM_ATTEMPTS)
        with pytest.raises(anthropic.APIError) as caught:
            call_llm(_config(), [{"role": "user", "content": "hi"}])
        assert "HTTP" not in classify_error(caught.value)[1]

    def test_a_context_window_stop_cuts_the_last_tool_call(self, monkeypatch, no_sleep):
        _serve(monkeypatch, [_message_events(_TOOL_USE, stop_reason="model_context_window_exceeded")])
        response = call_llm_with_tools(_config(), [{"role": "user", "content": "go"}], _TOOLS)
        assert response.tool_calls[0].parse_error

    def test_a_non_object_tool_input_is_not_replayed(self, monkeypatch, no_sleep):
        odd = {"type": "tool_use", "id": "t1", "name": "read_file", "input": [1, 2]}
        _serve(monkeypatch, [_message_events(_THINKING, odd, stop_reason="tool_use")])
        history = [{"role": "user", "content": "go"}]
        response = call_llm_with_tools(_config(), history, _TOOLS)
        assert response.tool_calls[0].parse_error
        history.append(assistant_tool_turn(response))
        history.append({"role": "tool", "tool_call_id": "t1", "content": "error"})
        kwargs = messages_call_kwargs(_config(), history, _TOOLS)
        (block,) = kwargs["messages"][1]["content"]
        assert block["input"] == {}

    def test_an_unreadable_event_mid_stream_is_retried(self, monkeypatch, no_sleep):
        calls: list = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                body = _sse(*_OK[:2]) + b"event: content_block_delta\ndata: {\"type\": \"cont\n\n"
            else:
                body = _sse(*_OK)
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

        real = anthropic.Anthropic
        monkeypatch.setattr(
            llm_anthropic,
            "Anthropic",
            lambda **kw: real(http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), **kw),
        )
        assert call_llm(_config(), [{"role": "user", "content": "hi"}]).content == "ok"
        assert len(calls) == 2

    def test_no_credential_at_all_is_an_auth_failure(self, monkeypatch, no_sleep):
        monkeypatch.setenv("HOME", "/nonexistent")
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent")
        _serve(monkeypatch, [_OK])
        config = LLMConfig(provider="Anthropic API", model_name="claude-haiku-5-5", api_key="")
        with pytest.raises(Exception) as caught:
            call_llm(config, [{"role": "user", "content": "hi"}])
        assert classify_error(caught.value)[0] == "auth"

    def test_a_config_without_the_display_name_still_finds_the_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        config = {"worker_provider_key": "Anthropic API", "worker_model": "claude-opus-5-5"}
        assert get_api_key(config, tier="worker") == "ant-key"


# ---------------------------------------------------------------------------
# Third review: fallback seams, thinking that can't go back, user turns
# ---------------------------------------------------------------------------

_FALLBACK = {"type": "fallback"}


def _with_fallback_block(events: list[dict], at: int) -> list[dict]:
    """Insert a fallback marker block at content index ``at``, renumbering."""
    out: list[dict] = []
    for event in events:
        event = json.loads(json.dumps(event))
        if "index" in event and event["index"] >= at:
            event["index"] += 1
        out.append(event)
    marker = [
        {"type": "content_block_start", "index": at,
         "content_block": {"type": "fallback", "from": {"model": "claude-opus-5-5"},
                           "to": {"model": "claude-opus-4-8"}}},
        {"type": "content_block_stop", "index": at},
    ]
    # Before the first event of the block that now sits at at + 1.
    position = next(
        i for i, e in enumerate(out)
        if e.get("type") == "content_block_start" and e.get("index") == at + 1
    )
    return [*out[:position], *marker, *out[position:]]


class TestThirdReview:
    def test_a_declined_models_tool_call_is_not_run(self, monkeypatch, no_sleep):
        declined = {"type": "tool_use", "id": "t0", "name": "read_file", "input": {"path": "x"}}
        events = _with_fallback_block(
            _message_events(declined, _TOOL_USE, stop_reason="tool_use"), at=1
        )
        _serve(monkeypatch, [events])
        response = call_llm_with_tools(_config(), [{"role": "user", "content": "go"}], _TOOLS)
        assert [tc.id for tc in response.tool_calls] == ["toolu_1"]

    def test_a_structured_answer_restarted_by_a_fallback_keeps_only_the_restart(
        self, monkeypatch, no_sleep
    ):
        events = _with_fallback_block(
            _message_events(
                {"type": "text", "text": '{"partial": '}, {"type": "text", "text": '{"ok": true}'}
            ),
            at=1,
        )
        _serve(monkeypatch, [events])
        config = _config(response_format={"type": "object", "properties": {"ok": {"type": "boolean"}}})
        assert call_llm(config, [{"role": "user", "content": "plan"}]).content == '{"ok": true}'

    @pytest.mark.parametrize(
        "blocks",
        [
            # Cut off before its signature arrived.
            [{"type": "thinking", "thinking": "hmm", "signature": ""}, _TOOL_USE],
            # Thinking as the turn's last block.
            [_TOOL_USE, _THINKING],
        ],
    )
    def test_thinking_the_api_would_refuse_is_not_replayed(self, monkeypatch, no_sleep, blocks):
        _serve(monkeypatch, [_message_events(*blocks, stop_reason="max_tokens")])
        history = [{"role": "user", "content": "go"}]
        response = call_llm_with_tools(_config(), history, _TOOLS)
        history.append(assistant_tool_turn(response))
        history.append({"role": "tool", "tool_call_id": "toolu_1", "content": "cut"})
        kwargs = messages_call_kwargs(_config(), history, _TOOLS)
        assert all(b["type"] != "thinking" for b in kwargs["messages"][1]["content"])

    def test_a_user_turn_after_tool_results_joins_them(self):
        _, out = messages_to_anthropic(
            [
                {"role": "user", "content": "go"},
                _tool_turn(),
                {"role": "tool", "tool_call_id": "toolu_1", "content": "result"},
                {"role": "user", "content": "Wrap up now."},
            ],
            model="m",
        )
        assert [m["role"] for m in out] == ["user", "assistant", "user"]
        assert out[2]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result"},
            {"type": "text", "text": "Wrap up now."},
        ]

    @pytest.mark.parametrize(
        "error", [httpx2.DecodingError("bad gzip"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")]
    )
    def test_an_undecodable_body_mid_stream_is_retried(self, monkeypatch, no_sleep, error):
        calls = _serve_raw(monkeypatch, [(_OK[:3], error), _OK])
        assert call_llm(_config(), [{"role": "user", "content": "hi"}]).content == "ok"
        assert len(calls) == 2

    def test_the_client_is_closed_and_connects_fast(self, monkeypatch, no_sleep):
        made: list = []
        real = anthropic.Anthropic

        def client(**kwargs):
            made.append(kwargs["timeout"])
            instance = real(
                http_client=httpx2.Client(transport=httpx2.MockTransport(
                    lambda r: httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                              content=_sse(*_OK))
                )),
                **kwargs,
            )
            made.append(instance)
            return instance

        monkeypatch.setattr(llm_anthropic, "Anthropic", client)
        call_llm(_config(), [{"role": "user", "content": "hi"}])
        timeout, instance = made
        assert timeout.connect == 10.0 and timeout.read == llm.call_timeout(_config())
        assert instance.is_closed()

    def test_a_mid_stream_auth_error_is_not_retried(self, monkeypatch, no_sleep):
        event = {"type": "error", "error": {"type": "authentication_error", "message": "x"}}
        calls = _serve(monkeypatch, [[*_OK[:1], event]])
        with pytest.raises(anthropic.APIError):
            call_llm(_config(), [{"role": "user", "content": "hi"}])
        assert len(calls) == 1

    def test_a_stream_past_the_call_timeout_is_cut(self, monkeypatch, no_sleep):
        # Every reading is ten seconds after the last: the events keep coming,
        # but the whole stream outlasts the five-second limit.
        clock = itertools.count(step=10.0)
        monkeypatch.setattr(llm_anthropic.time, "monotonic", lambda: next(clock))
        calls = _serve(monkeypatch, [_OK] * llm.STREAM_ATTEMPTS)
        with pytest.raises(anthropic.APITimeoutError):
            call_llm(_config(timeout=5), [{"role": "user", "content": "hi"}])
        assert len(calls) == llm.STREAM_ATTEMPTS

    def test_an_architect_on_another_provider_never_gets_the_workers_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        config = {"worker_provider_key": "Anthropic API", "architect_provider_key": "OpenAI API"}
        assert get_api_key(config, tier="architect") == ""

    def test_an_architect_on_the_same_provider_shares_the_workers_key(self, monkeypatch):
        monkeypatch.setenv("STRIDE_GPT_API_KEY", "sk-generic")
        config = {"worker_provider_key": "OpenAI API", "architect_provider_key": "OpenAI API"}
        assert get_api_key(config, tier="architect") == "sk-generic"


def test_cli_error_lines_do_not_wrap():
    """vega's runner reads the CLI's failure lines one at a time."""
    import io

    from rich.console import Console

    from stride_gpt import cli
    from stride_gpt.agent.progress import RichProgress

    buffer = io.StringIO()
    console = Console(file=buffer, width=80)
    RichProgress(console).error("auth-service", "x " * 100)
    original = cli.console
    cli.console = console
    try:
        error = anthropic.RateLimitError(
            "slow", response=httpx2.Response(429, request=httpx2.Request("POST", "http://x")), body=None
        )
        cli._report_provider_failure(error)
    finally:
        cli.console = original
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert len(lines) == 3, lines


class TestCoverageGaps:
    def test_redacted_thinking_is_replayed_as_produced(self, monkeypatch, no_sleep):
        events = _message_events(_TOOL_USE, stop_reason="tool_use")
        redacted = [
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "redacted_thinking", "data": "opaque"}},
            {"type": "content_block_stop", "index": 0},
        ]
        for event in events:
            if "index" in event:
                event["index"] += 1
        _serve(monkeypatch, [[events[0], *redacted, *events[1:]]])
        history = [{"role": "user", "content": "go"}]
        history.append(assistant_tool_turn(call_llm_with_tools(_config(), history, _TOOLS)))
        history.append({"role": "tool", "tool_call_id": "toolu_1", "content": "x"})
        kwargs = messages_call_kwargs(_config(), history, _TOOLS)
        assert kwargs["messages"][1]["content"] == [
            {"type": "redacted_thinking", "data": "opaque"},
            _TOOL_USE,
        ]

    def test_three_rounds_keep_every_turns_thinking(self, monkeypatch, no_sleep):
        rounds = [
            _message_events(
                {"type": "thinking", "thinking": f"step {n}", "signature": f"sig-{n}"},
                {"type": "tool_use", "id": f"t{n}", "name": "read_file", "input": {"path": f"{n}.py"}},
                stop_reason="tool_use",
            )
            for n in range(3)
        ]
        calls = _serve(monkeypatch, rounds)
        history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
        for n in range(3):
            history.append(assistant_tool_turn(call_llm_with_tools(_config(), history, _TOOLS)))
            history.append({"role": "tool", "tool_call_id": f"t{n}", "content": "ok"})
        sent = _body(calls[2])["messages"]
        signatures = [b.get("signature") for m in sent for b in m["content"] if isinstance(b, dict)
                      and b.get("type") == "thinking"]
        assert signatures == ["sig-0", "sig-1"]

    def test_an_image_call_goes_over_the_messages_api(self, monkeypatch, no_sleep):
        from stride_gpt.core.llm import call_llm_with_image

        calls = _serve(monkeypatch, [_OK])
        assert call_llm_with_image(_config(), "what is this", "AAAA", "image/png").content == "ok"
        image = next(
            b for m in _body(calls[0])["messages"] for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "image"
        )
        assert image["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}


class TestFourthReview:
    def _printed(self, exc) -> list[str]:
        import io

        from rich.console import Console

        from stride_gpt import cli

        buffer = io.StringIO()
        original = cli.console
        cli.console = Console(file=buffer, width=80)
        try:
            assert cli._report_provider_failure(exc)
        finally:
            cli.console = original
        return [line for line in buffer.getvalue().splitlines() if line.strip()]

    def _status_error(self, status: int, body: dict):
        response = httpx2.Response(status, request=httpx2.Request("POST", "http://p/v1/messages"))
        cls = {403: anthropic.PermissionDeniedError, 502: anthropic.InternalServerError}[status]
        return cls("x", response=response, body=body)

    def test_brackets_in_a_providers_body_are_printed_not_parsed(self):
        lines = self._printed(self._status_error(502, {"detail": "bad gateway [/v1/messages] [cost_cap]"}))
        assert "[/v1/messages] [cost_cap]" in lines[0]

    def test_a_multi_line_body_prints_as_one_line(self):
        lines = self._printed(
            self._status_error(403, {"detail": "insufficient balance\nscan cost cap exceeded"})
        )
        assert len(lines) == 1
        assert "(HTTP 403: insufficient balance scan cost cap exceeded)" in lines[0]

    def test_a_synthesis_failure_reason_does_not_wrap(self):
        import io

        from rich.console import Console

        from stride_gpt.agent.progress import RichProgress

        buffer = io.StringIO()
        RichProgress(Console(file=buffer, width=80)).synthesis_failed("x " * 100)
        assert len([line for line in buffer.getvalue().splitlines() if line.strip()]) == 2

    def test_thinking_left_last_by_a_dropped_empty_text_is_not_replayed(self, monkeypatch, no_sleep):
        blocks = [_TOOL_USE, _THINKING, {"type": "text", "text": ""}]
        _serve(monkeypatch, [_message_events(*blocks, stop_reason="max_tokens")])
        history = [{"role": "user", "content": "go"}]
        history.append(assistant_tool_turn(call_llm_with_tools(_config(), history, _TOOLS)))
        history.append({"role": "tool", "tool_call_id": "toolu_1", "content": "cut"})
        kwargs = messages_call_kwargs(_config(), history, _TOOLS)
        assert all(b["type"] != "thinking" for b in kwargs["messages"][1]["content"])

    def test_the_planners_schema_becomes_a_structured_output(self):
        from stride_gpt.core.schemas import AnalysisPlan

        kwargs = messages_call_kwargs(
            _config(response_format=AnalysisPlan.model_json_schema()),
            [{"role": "user", "content": "plan"}],
        )
        schema = kwargs["output_config"]["format"]["schema"]
        assert schema["additionalProperties"] is False
        assert all(d.get("additionalProperties") is False for d in schema.get("$defs", {}).values()
                   if d.get("type") == "object")

"""Retries on provider errors in stride_gpt.core.llm (issue #255).

A request that fails before its response starts is retried by the OpenAI SDK
(``max_retries``); a stream that dies after it started is retried by
``_stream_response``. These run the real client against a mock transport, so
they check what actually goes over the wire.
"""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from stride_gpt.core import llm
from stride_gpt.core.llm import call_llm
from stride_gpt.core.schemas import LLMConfig

_MESSAGES = [{"role": "user", "content": "hi"}]
_OK = {
    "id": "resp_1",
    "object": "response",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
}


def _config() -> LLMConfig:
    return LLMConfig(provider="OpenAI API", model_name="gpt-6-sol", api_key="sk-test")


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr("openai._base_client.time.sleep", slept.append)
    return slept


def _sse(*events: dict) -> bytes:
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events
    )


_COMPLETED = {"type": "response.completed", "sequence_number": 1, "response": _OK}
_CREATED = {"type": "response.created", "sequence_number": 0, "response": {**_OK, "output": []}}


class Stream:
    """A 200 event stream: ``events``, then ``then`` raised mid-body if set."""

    def __init__(self, *events: dict, then: Exception | None = None):
        self.events = events
        self.then = then

    def body(self):
        yield _sse(*self.events)
        if self.then is not None:
            raise self.then


def _script(monkeypatch, outcomes: list) -> list[httpx.Request]:
    """Serve ``outcomes`` in order: (status, headers) or (status, headers, body),
    a Stream, or an exception to raise. A 200 is a stream that completes."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, Stream):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=outcome.body()
            )
        status, headers, *body = outcome
        if status == 200:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(_COMPLETED)
            )
        return httpx.Response(
            status, headers=headers, json=body[0] if body else {"error": {"message": "boom"}}
        )

    real = llm.OpenAI

    def client(**kwargs):
        return real(http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(llm, "OpenAI", client)
    return calls


_SUCCESS = (200, {})


class TestRetries:
    @pytest.mark.parametrize("status", [408, 429, 500, 503])
    def test_transient_error_is_retried_and_call_succeeds(self, monkeypatch, sleeps, status):
        calls = _script(monkeypatch, [(status, {}), _SUCCESS])
        assert call_llm(_config(), _MESSAGES).content == "ok"
        assert len(calls) == 2
        assert len(sleeps) == 1

    def test_connection_error_is_retried(self, monkeypatch, sleeps):
        calls = _script(monkeypatch, [httpx.ConnectError("reset"), _SUCCESS])
        assert call_llm(_config(), _MESSAGES).content == "ok"
        assert len(calls) == 2

    def test_auth_error_is_not_retried(self, monkeypatch, sleeps):
        calls = _script(monkeypatch, [(401, {})])
        with pytest.raises(openai.AuthenticationError):
            call_llm(_config(), _MESSAGES)
        assert len(calls) == 1
        assert sleeps == []

    def test_gives_up_after_the_sdk_retries(self, monkeypatch, sleeps):
        calls = _script(monkeypatch, [(429, {}) for _ in range(llm.SDK_RETRIES + 1)])
        with pytest.raises(openai.RateLimitError):
            call_llm(_config(), _MESSAGES)
        assert len(calls) == llm.SDK_RETRIES + 1
        assert len(sleeps) == llm.SDK_RETRIES

    def test_retry_after_header_sets_the_delay(self, monkeypatch, sleeps):
        _script(monkeypatch, [(429, {"retry-after": "17"}), _SUCCESS])
        call_llm(_config(), _MESSAGES)
        assert sleeps == [17.0]


@pytest.fixture
def backoff(monkeypatch) -> list[float]:
    """The waits ``_stream_response`` took between attempts."""
    slept: list[float] = []
    monkeypatch.setattr(llm, "_sleep", slept.append)
    return slept


class TestStreams:
    def test_every_call_streams(self, monkeypatch, sleeps):
        calls = _script(monkeypatch, [_SUCCESS])
        assert call_llm(_config(), _MESSAGES).content == "ok"
        assert json.loads(calls[0].content)["stream"] is True

    def test_a_stream_that_drops_mid_body_is_retried(self, monkeypatch, sleeps, backoff):
        calls = _script(
            monkeypatch, [Stream(_CREATED, then=httpx.ReadError("reset")), _SUCCESS]
        )
        assert call_llm(_config(), _MESSAGES).content == "ok"
        assert len(calls) == 2
        assert backoff == [2]

    def test_a_stream_that_ends_without_completing_is_retried(self, monkeypatch, sleeps, backoff):
        calls = _script(monkeypatch, [Stream(_CREATED), _SUCCESS])
        assert call_llm(_config(), _MESSAGES).content == "ok"
        assert len(calls) == 2

    def test_a_gateway_request_timeout_event_is_retried(self, monkeypatch, sleeps, backoff):
        died = {"type": "error", "sequence_number": 1, "code": "request_timeout",
                "message": "stream closed before response.completed"}
        calls = _script(monkeypatch, [Stream(_CREATED, died), _SUCCESS])
        assert call_llm(_config(), _MESSAGES).content == "ok"
        assert len(calls) == 2

    def test_a_gateway_request_timeout_400_is_retried(self, monkeypatch, sleeps, backoff):
        body = {"error": {"message": "stream closed before response.completed",
                          "code": "request_timeout"}}
        calls = _script(monkeypatch, [(400, {}, body), _SUCCESS])
        assert call_llm(_config(), _MESSAGES).content == "ok"
        assert len(calls) == 2

    def test_a_failed_response_is_not_retried(self, monkeypatch, sleeps, backoff):
        failed = {"type": "response.failed", "sequence_number": 1, "response": {
            **_OK, "status": "failed", "output": [],
            "error": {"code": "invalid_prompt", "message": "refused"}}}
        calls = _script(monkeypatch, [Stream(_CREATED, failed)])
        with pytest.raises(openai.APIError, match="refused") as raised:
            call_llm(_config(), _MESSAGES)
        assert raised.value.code == "invalid_prompt"
        assert len(calls) == 1
        assert backoff == []

    def test_gives_up_after_three_dead_streams(self, monkeypatch, sleeps, backoff):
        calls = _script(monkeypatch, [Stream(_CREATED) for _ in range(3)])
        with pytest.raises(openai.APIError, match="ended before the response completed"):
            call_llm(_config(), _MESSAGES)
        assert len(calls) == 3
        assert backoff == [2, 4]

    def test_a_connection_error_before_the_stream_is_left_to_the_sdk(
        self, monkeypatch, sleeps, backoff
    ):
        tries = llm.SDK_RETRIES + 1
        calls = _script(monkeypatch, [httpx.ConnectError("refused") for _ in range(tries)])
        with pytest.raises(openai.APIConnectionError):
            call_llm(_config(), _MESSAGES)
        assert len(calls) == tries  # the SDK's own attempts, not multiplied here
        assert backoff == []

    def test_a_stream_past_the_call_timeout_is_cut(self, monkeypatch, sleeps, backoff):
        clock = iter(range(0, 10_000, 700))
        monkeypatch.setattr("stride_gpt.core.llm.time.monotonic", lambda: next(clock))
        config = _config().model_copy(update={"timeout": 1000})
        calls = _script(monkeypatch, [Stream(_CREATED, _COMPLETED) for _ in range(3)])
        with pytest.raises(openai.APITimeoutError):
            call_llm(config, _MESSAGES)
        assert len(calls) == 3


_REASONING = {"type": "reasoning", "id": "rs_1", "status": "completed",
              "summary": [], "encrypted_content": "opaque"}


class TestReasoning:

    def test_encrypted_reasoning_is_asked_for_and_returned(self, monkeypatch, sleeps):
        completed = {**_COMPLETED, "response": {**_OK, "output": [_REASONING, *_OK["output"]]}}
        calls = _script(monkeypatch, [Stream(completed)])
        response = call_llm(_config(), _MESSAGES)
        assert json.loads(calls[0].content)["include"] == ["reasoning.encrypted_content"]
        assert response.reasoning_items == [
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"}
        ]


def _message(content: list[dict]) -> dict:
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": content}


class TestTerminalStates:
    def test_the_stream_is_read_to_its_end(self, monkeypatch, sleeps):
        """Hanging up at response.completed made the proxy record a failure."""
        consumed = []

        class Tracked(Stream):
            def body(self):
                yield from super().body()
                consumed.append(True)

        _script(monkeypatch, [Tracked(_CREATED, _COMPLETED)])
        assert call_llm(_config(), _MESSAGES).content == "ok"
        assert consumed == [True]

    def test_an_incomplete_response_is_marked(self, monkeypatch, sleeps, caplog):
        incomplete = {"type": "response.incomplete", "sequence_number": 1, "response": {
            **_OK, "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"}}}
        _script(monkeypatch, [Stream(_CREATED, incomplete)])
        response = call_llm(_config(), _MESSAGES)
        assert response.incomplete_reason == "max_output_tokens"
        assert "incomplete" in caplog.text

    def test_a_completed_response_is_not_marked(self, monkeypatch, sleeps):
        _script(monkeypatch, [_SUCCESS])
        assert call_llm(_config(), _MESSAGES).incomplete_reason is None

    def test_a_content_filter_stop_is_raised(self, monkeypatch, sleeps):
        incomplete = {"type": "response.incomplete", "sequence_number": 1, "response": {
            **_OK, "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"}}}
        _script(monkeypatch, [Stream(_CREATED, incomplete)])
        with pytest.raises(llm.ModelRefusalError, match="content filter"):
            call_llm(_config(), _MESSAGES)

    def test_a_refusal_is_raised_not_returned_blank(self, monkeypatch, sleeps):
        refused = {**_COMPLETED, "response": {**_OK, "output": [
            _message([{"type": "refusal", "refusal": "I can't help with that."}])]}}
        _script(monkeypatch, [Stream(refused)])
        with pytest.raises(llm.ModelRefusalError, match="can't help"):
            call_llm(_config(), _MESSAGES)

    def test_a_gateway_upstream_failure_mid_stream_is_retried(self, monkeypatch, sleeps, backoff):
        failed = {"type": "response.failed", "sequence_number": 1, "response": {
            **_OK, "status": "failed", "output": [],
            "error": {"code": "upstream_error", "message": "Upstream request failed"}}}
        calls = _script(monkeypatch, [Stream(_CREATED, failed), _SUCCESS])
        assert call_llm(_config(), _MESSAGES).content == "ok"
        assert len(calls) == 2

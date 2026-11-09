"""Retries on provider errors in stride_gpt.core.llm (issue #255).

Retrying belongs to the OpenAI SDK (``max_retries``). These run the real client
against a mock transport, so they check what actually goes over the wire.
"""

from __future__ import annotations

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


def _script(monkeypatch, outcomes: list) -> list[httpx.Request]:
    """Serve ``outcomes`` in order: (status, headers), or an exception to raise."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        status, headers = outcome
        body = _OK if status == 200 else {"error": {"message": "boom"}}
        return httpx.Response(status, headers=headers, json=body)

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

    def test_gives_up_after_three_retries(self, monkeypatch, sleeps):
        calls = _script(monkeypatch, [(429, {}) for _ in range(4)])
        with pytest.raises(openai.RateLimitError):
            call_llm(_config(), _MESSAGES)
        assert len(calls) == 4
        assert len(sleeps) == 3

    def test_retry_after_header_sets_the_delay(self, monkeypatch, sleeps):
        _script(monkeypatch, [(429, {"retry-after": "17"}), _SUCCESS])
        call_llm(_config(), _MESSAGES)
        assert sleeps == [17.0]

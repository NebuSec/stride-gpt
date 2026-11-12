"""Tests for stride_gpt.agent.errors -- classifying a subsystem failure."""

from __future__ import annotations

import openai
import pytest

from stride_gpt.agent.errors import classify_error
from tests.fakes import connection_error, status_error


class TestClassifyByType:
    @pytest.mark.parametrize(
        ("exc_class", "status", "expected"),
        [
            (openai.RateLimitError, 429, "rate_limited"),
            (openai.AuthenticationError, 401, "auth"),
            (openai.PermissionDeniedError, 403, "auth"),
            (openai.InternalServerError, 500, "provider_error"),
            (openai.BadRequestError, 400, "provider_error"),
        ],
    )
    def test_each_sdk_error_maps_to_its_class(self, exc_class, status, expected):
        error_class, reason = classify_error(status_error(exc_class, status=status))
        assert error_class == expected
        # The reason is what the console and the report show, so it has to
        # carry the provider's own wording too.
        assert "boom" in reason

    def test_a_connection_error_is_a_provider_error(self):
        assert classify_error(connection_error())[0] == "provider_error"

    def test_the_type_wins_over_a_misleading_message(self):
        """A rate limit that happens to mention a crash is still a rate limit."""
        exc = status_error(openai.RateLimitError, message="the worker crashed", status=429)
        assert classify_error(exc)[0] == "rate_limited"


class TestContextOverflow:
    def test_the_error_code_identifies_it(self):
        exc = status_error(
            openai.BadRequestError,
            message="too big",
            body={"code": "context_length_exceeded", "message": "too big"},
        )
        assert classify_error(exc)[0] == "context_overflow"

    @pytest.mark.parametrize(
        "message",
        [
            "This model's maximum context length is 1050000 tokens.",
            "Your input exceeds the context window of this model.",
        ],
    )
    def test_the_wording_identifies_it(self, message):
        exc = status_error(openai.BadRequestError, message=message)
        assert classify_error(exc)[0] == "context_overflow"

    def test_another_bad_request_is_not_an_overflow(self):
        exc = status_error(openai.BadRequestError, message="unknown parameter")
        assert classify_error(exc)[0] == "provider_error"


class TestClassifyByStatus:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [(429, "rate_limited"), (401, "auth"), (403, "auth"), (503, "provider_error")],
    )
    def test_a_wrapped_error_is_classified_by_its_status(self, status, expected):
        exc = RuntimeError("gateway said no")
        exc.status_code = status
        assert classify_error(exc)[0] == expected

    def test_a_status_we_do_not_recognise_falls_through(self):
        exc = RuntimeError("teapot")
        exc.status_code = 418
        assert classify_error(exc)[0] == "unexpected"

    def test_a_non_numeric_status_is_ignored(self):
        exc = RuntimeError("weird")
        exc.status_code = "not a number"
        assert classify_error(exc)[0] == "unexpected"


class TestClassifyByMessage:
    """The last resort — local runtimes that report failures as plain strings."""

    def test_llama_cpp_context_overflow(self):
        exc = RuntimeError("n_keep = 4 exceeds n_ctx = 4096")
        error_class, reason = classify_error(exc)
        assert error_class == "context_overflow"
        assert "Context window exceeded" in reason

    def test_a_crashed_local_model(self):
        error_class, reason = classify_error(RuntimeError("Model has crashed"))
        assert error_class == "provider_error"
        assert "memory constraints" in reason

    @pytest.mark.parametrize(
        ("exc_class", "status"),
        [(openai.BadRequestError, 400), (openai.InternalServerError, 500)],
    )
    def test_llama_cpp_context_overflow_through_the_sdk(self, exc_class, status):
        """A compatible endpoint's runtime reaches us as a generic SDK exception."""
        exc = status_error(exc_class, message="n_keep: 5000 >= n_ctx: 4096", status=status)
        error_class, reason = classify_error(exc)
        assert error_class == "context_overflow"
        assert "n_ctx" in reason

    @pytest.mark.parametrize(
        ("exc_class", "status", "expected"),
        [
            (openai.RateLimitError, 429, "rate_limited"),
            (openai.AuthenticationError, 401, "auth"),
        ],
    )
    def test_the_text_never_overrides_a_specific_class(self, exc_class, status, expected):
        exc = status_error(exc_class, message="n_keep and n_ctx in passing", status=status)
        assert classify_error(exc)[0] == expected

    def test_one_of_the_two_markers_is_not_an_overflow(self):
        exc = status_error(openai.BadRequestError, message="bad n_keep value")
        assert classify_error(exc)[0] == "provider_error"

    def test_anything_else_is_unexpected(self):
        error_class, reason = classify_error(ValueError("something odd"))
        assert error_class == "unexpected"
        assert reason == "Unexpected error: something odd"


class TestProviderMessage:
    """The reason quotes the provider's own message, not the SDK's wrapping (#305)."""

    def test_the_parsed_body_is_preferred_and_the_request_id_dropped(self):
        exc = status_error(
            openai.AuthenticationError,
            message="Error code: 401 - {'message': 'Bad key (request_id: 0a1b-2c3d)'}",
            status=401,
            body={"message": "Bad key (request_id: 0a1b-2c3d)", "type": "invalid_request_error"},
        )
        assert classify_error(exc) == (
            "auth",
            "The provider rejected the credentials for this model. (Bad key)",
        )

    def test_a_nested_error_message_is_extracted(self):
        exc = status_error(
            openai.AuthenticationError,
            status=401,
            body={"error": {"type": "authentication_error", "message": "invalid x-api-key"}},
        )
        assert classify_error(exc)[1].endswith("(invalid x-api-key)")

    def test_a_json_text_is_read_when_there_is_no_body(self):
        exc = status_error(
            openai.InternalServerError,
            message='Error code: 500 - {"error": {"message": "upstream died"}}',
            status=500,
        )
        assert classify_error(exc)[1].endswith("(upstream died)")

    def test_a_plain_text_error_keeps_its_text(self):
        exc = connection_error()
        assert classify_error(exc)[1] == "The provider failed to answer. (Connection error.)"


class TestClassifyStreamedFailures:
    """A failure inside a started stream carries a code, not an HTTP status."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("context_length_exceeded", "context_overflow"),
            ("rate_limit_exceeded", "rate_limited"),
            ("invalid_api_key", "auth"),
            ("upstream_error", "provider_error"),
        ],
    )
    def test_by_code(self, code, expected):
        import httpx

        exc = openai.APIError("boom", httpx.Request("POST", "http://x"), body={"code": code})
        assert classify_error(exc)[0] == expected

    def test_a_refusal(self):
        from stride_gpt.core.llm import ModelRefusalError

        error_class, reason = classify_error(ModelRefusalError("not this one"))
        assert error_class == "provider_error"
        assert "declined" in reason and "not this one" in reason

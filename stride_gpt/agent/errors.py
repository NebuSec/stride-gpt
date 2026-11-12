"""Classify a subsystem failure from the exception, not from its message.

The agent loop used to decide what went wrong by searching the error text for
``"n_keep"`` or ``"crashed"``. That only ever recognised two local-model
failures, and it silently reclassified anything whose wording changed. Every
provider call goes through the OpenAI SDK, which raises typed exceptions carrying
an HTTP status, so the type is the reliable signal and the text is the fallback.
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any

import openai

from stride_gpt.core.llm import ModelRefusalError
from stride_gpt.core.schemas import ErrorClass

_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "exceeds the context window",
)

_REASONS: dict[ErrorClass, str] = {
    "context_overflow": (
        "Context window exceeded — model ran out of space for this subsystem."
    ),
    "rate_limited": "Rate limited by the provider.",
    "auth": "The provider rejected the credentials for this model.",
    "provider_error": "The provider failed to answer.",
}


def _is_context_overflow(exc: openai.BadRequestError) -> bool:
    """The SDK has no overflow class: it is a 400 told apart by code or wording."""
    if getattr(exc, "code", None) == "context_length_exceeded":
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _CONTEXT_OVERFLOW_MARKERS)


def _by_type(exc: BaseException) -> ErrorClass | None:
    """Map an OpenAI SDK exception class to an error class."""
    if isinstance(exc, openai.BadRequestError) and _is_context_overflow(exc):
        return "context_overflow"
    if isinstance(exc, openai.RateLimitError):
        return "rate_limited"
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return "auth"
    return None


# Codes a Responses stream reports in an `error` or `response.failed` event,
# which carries no HTTP status: the stream had already started with a 200.
_CODE_CLASSES: dict[str, ErrorClass] = {
    "context_length_exceeded": "context_overflow",
    "rate_limit_exceeded": "rate_limited",
    "insufficient_quota": "rate_limited",
    "invalid_api_key": "auth",
    "authentication_failed": "auth",
    "permission_denied": "auth",
}


def _by_code(exc: BaseException) -> ErrorClass | None:
    """Map the error code a streamed failure carries to an error class."""
    code = getattr(exc, "code", None)
    return _CODE_CLASSES.get(code) if isinstance(code, str) else None


def _by_provider_error(exc: BaseException) -> ErrorClass | None:
    """Whether ``exc`` is a failed provider call of some unclassified kind."""
    # Connection errors, timeouts, 5xx, and malformed requests.
    return "provider_error" if isinstance(exc, openai.APIError) else None


def _by_status(exc: BaseException) -> ErrorClass | None:
    """Map an HTTP status carried on the exception, whatever raised it."""
    status: Any = getattr(exc, "status_code", None)
    try:
        code = int(status)
    except (TypeError, ValueError):
        return None
    if code == 429:
        return "rate_limited"
    if code in (401, 403):
        return "auth"
    if 500 <= code < 600:
        return "provider_error"
    return None


# The SDK renders a failed call as "Error code: 401 - {<python repr of the body>}".
_STATUS_PREFIX = re.compile(r"^Error code: \d+ - ")
_REQUEST_ID = re.compile(r"\s*\(request[_ ]id:[^)]*\)", re.IGNORECASE)


def _message_from(body: object) -> str | None:
    """The human-readable message in an error body, if it has one."""
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
        if isinstance(body.get("message"), str):
            return body["message"]
    return None


def provider_message(exc: BaseException) -> str:
    """The provider's own error message, without the SDK's wrapping.

    Prefers the parsed body on the exception. Falls back to the exception text,
    minus the status prefix, read as JSON when it is. Request ids are dropped:
    they mean nothing to the user and make the line long.
    """
    text = _message_from(getattr(exc, "body", None))
    if text is None:
        text = _STATUS_PREFIX.sub("", str(exc), count=1)
        with contextlib.suppress(ValueError):
            text = _message_from(json.loads(text)) or text
    return _REQUEST_ID.sub("", text).strip()


def classify_error(exc: BaseException) -> tuple[ErrorClass, str]:
    """Return the error class for ``exc`` and the reason to show the user."""
    err_str = str(exc)

    if isinstance(exc, ModelRefusalError):
        return "provider_error", f"The model declined to answer: {err_str}"

    # Status before the generic provider check: a status on a wrapped exception
    # is more specific than "some API error".
    error_class = _by_type(exc) or _by_status(exc) or _by_code(exc) or _by_provider_error(exc)

    # llama.cpp says "n_keep"/"n_ctx" when the prompt won't fit. That arrives as
    # a plain string, or wrapped in a generic SDK exception with whatever
    # status the runtime chose, so the text refines a generic failure. It never
    # overrides a specific class.
    if error_class in (None, "provider_error") and "n_keep" in err_str and "n_ctx" in err_str:
        error_class = "context_overflow"

    if error_class is not None:
        reason = _REASONS[error_class]
        message = provider_message(exc)
        return error_class, f"{reason} ({message})" if message else reason

    # Last resort — a provider that wraps its errors, or a local runtime that
    # reports them as plain strings. LM Studio says the model "crashed" when it
    # is killed for memory.
    if "crashed" in err_str.lower():
        return "provider_error", "The model crashed, likely due to memory constraints."

    return "unexpected", f"Unexpected error: {err_str}"

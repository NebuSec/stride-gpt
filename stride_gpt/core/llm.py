"""Unified LLM interface: the OpenAI SDK against the Responses API.

Agent history stays in Chat Completions shape and is converted at this boundary.
"""

from __future__ import annotations

import json as _json
import logging
import time

import openai
from openai import OpenAI

from stride_gpt.core.schemas import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    LLMConfig,
    LLMResponse,
    ToolCallResult,
)
from stride_gpt.models import get_model, get_provider_by_key, model_uses_completion_tokens

logger = logging.getLogger(__name__)

# Attempts at one call whose stream died. The SDK's own retries cover a request
# that failed before its response started; nothing else retries one that died
# mid-stream, and a failed call costs the agent loop a whole subsystem.
STREAM_ATTEMPTS = 3
# Error codes that mean the stream died, not that the request was wrong. A
# gateway that cuts a stalled stream reports request_timeout, as a 400 or an
# error event; the same request then usually finishes in seconds.
_DEAD_STREAM_CODES = frozenset({"request_timeout", "server_error"})
_REASONING_INCLUDE = "reasoning.encrypted_content"
_sleep = time.sleep  # the backoff between stream attempts; tests replace it


def _output_token_cap(config: LLMConfig) -> int | None:
    """``max_output_tokens`` for a call, or None to leave the model's own limit.

    A model the registry doesn't know (a gateway route, a newer model) gets no
    cap, as it did before this fork replaced LiteLLM: a guessed one cut long
    answers short, and an incomplete answer reads like a finished one.
    """
    if config.max_tokens:
        return config.max_tokens
    registered = get_model(config.provider, config.model_name)
    return registered.default_tokens if registered is not None else None


def call_llm(config: LLMConfig, messages: list[dict]) -> LLMResponse:
    """Call an LLM with the given messages."""
    return _call_responses(config, messages)


def call_llm_with_tools(
    config: LLMConfig, messages: list[dict], tools: list[dict]
) -> LLMResponse:
    """Call an LLM with tool definitions. Returns response with optional tool_calls."""
    return _call_responses(config, messages, tools=tools)


def call_llm_with_image(
    config: LLMConfig,
    prompt: str,
    base64_image: str,
    media_type: str = "image/jpeg",
) -> LLMResponse:
    """Call an LLM with an image for analysis."""
    return _call_responses(config, _image_messages(config, prompt, base64_image, media_type))


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def _client_kwargs(config: LLMConfig) -> dict:
    """Kwargs for OpenAI() -- api_key, base_url, retries, timeout.

    A saved config can still name a provider this build dropped. Refuse it here
    rather than send that request, and possibly that key, to the wrong service.
    """
    if get_provider_by_key(config.provider) is None:
        raise ValueError(
            f"Unsupported provider {config.provider!r}: only OpenAI is available. "
            "Run /config to choose an OpenAI model."
        )
    # The SDK retries 408, 429, 5xx and dropped connections with exponential
    # backoff, and honours Retry-After. The agent loop bursts many calls in
    # close succession and easily trips TPM caps on tight org tiers.
    kwargs: dict = {
        "api_key": config.api_key or None,
        "max_retries": 3,
        "timeout": call_timeout(config),
    }
    if config.api_base:
        kwargs["base_url"] = config.api_base.rstrip("/")
    return kwargs


def call_timeout(config: LLMConfig) -> int:
    """Seconds one call may take; see DEFAULT_CALL_TIMEOUT_SECONDS."""
    return config.timeout if config.timeout and config.timeout > 0 else DEFAULT_CALL_TIMEOUT_SECONDS


def _openai_client(config: LLMConfig) -> OpenAI:
    return OpenAI(**_client_kwargs(config))


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


def _responses_text_format(config: LLMConfig) -> dict | None:
    """Responses API `text.format` payload, or None to omit."""
    if isinstance(config.response_format, dict):
        # strict is sent as False on purpose. Left out, the API now treats it as
        # True, and strict mode requires `additionalProperties: false` on every
        # object, which a Pydantic model_json_schema() does not carry: the
        # planner's AnalysisPlan request is refused as an invalid schema.
        return {
            "format": {
                "type": "json_schema",
                "name": "response",
                "schema": config.response_format,
                "strict": False,
            }
        }
    if config.response_format == "json":
        return {"format": {"type": "json_object"}}
    return None


def _to_responses_tools(tools: list[dict]) -> list[dict]:
    """Chat Completions nested tools → Responses flat function tools."""
    converted: list[dict] = []
    for tool in tools:
        fn = tool.get("function") if tool.get("type") == "function" else None
        if isinstance(fn, dict):
            converted.append(
                {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    # A required field. Left out, the API applies strict mode,
                    # which these schemas (no additionalProperties: false) are
                    # not written for.
                    "strict": bool(fn.get("strict", False)),
                }
            )
        else:
            converted.append(tool)
    return converted


def _convert_content_parts(content: object) -> object:
    """Chat Completions content parts → Responses input_text / input_image."""
    if not isinstance(content, list):
        return content
    converted: list = []
    for part in content:
        if not isinstance(part, dict):
            converted.append(part)
            continue
        ptype = part.get("type")
        if ptype == "text":
            converted.append({"type": "input_text", "text": part.get("text", "")})
        elif ptype == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            detail = (image.get("detail") if isinstance(image, dict) else None) or "auto"
            # detail is a required field of input_image.
            converted.append({"type": "input_image", "image_url": url, "detail": detail})
        else:
            converted.append(part)
    return converted


def messages_to_responses_input(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split Chat Completions-shaped messages into (instructions, input items).

    Agent loops keep the Chat Completions history format; this is the Responses
    boundary. System messages become `instructions`. Tool results become
    `function_call_output` items.
    """
    instructions_parts: list[str] = []
    items: list[dict] = []
    leading = True

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        # Only the system messages that open the conversation are its
        # instructions; one that arrives later stays where it was said.
        if role == "system" and leading and isinstance(content, str):
            if content:
                instructions_parts.append(content)
            continue
        leading = False

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or "",
                    "output": content if isinstance(content, str) else _json.dumps(content),
                }
            )
            continue

        if role == "assistant" and msg.get("tool_calls"):
            # The reasoning behind these calls goes back before them, as the
            # model produced it: without it the model starts every round of a
            # tool loop with no memory of why it made its previous calls.
            items.extend(msg.get("reasoning_items") or [])
            if isinstance(content, str) and content:
                items.append({"role": "assistant", "content": content})
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                arguments = fn.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = _json.dumps(arguments)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id") or tc.get("call_id") or "",
                        "name": fn.get("name") or "",
                        "arguments": arguments,
                    }
                )
            continue

        item: dict = {"role": role or "user", "content": _convert_content_parts(content)}
        items.append(item)

    instructions = "\n\n".join(instructions_parts) or None
    if not items and instructions:
        # The API refuses an empty input: a system-only request is sent as
        # input instead of instructions.
        items = [{"role": "system", "content": instructions}]
        instructions = None
    return instructions, items


def _responses_call_kwargs(
    config: LLMConfig,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """Kwargs for client.responses.create()."""
    instructions, input_items = messages_to_responses_input(messages)
    text_format = None if tools else _responses_text_format(config)
    if (
        instructions
        and text_format is not None
        and text_format["format"]["type"] == "json_object"
        and not _mentions_json(input_items)
    ):
        # JSON mode requires the word "json" in `input`; `instructions` does
        # not count. A prompt that asks for JSON only in its system message
        # would be refused, so that message goes first in `input` instead.
        input_items = [{"role": "system", "content": instructions}, *input_items]
        instructions = None
    kwargs: dict = {
        "model": config.model_name,
        "input": input_items,
        "store": False,
    }
    cap = _output_token_cap(config)
    if cap is not None:
        kwargs["max_output_tokens"] = cap
    if instructions:
        kwargs["instructions"] = instructions
    if _keeps_reasoning(config):
        # store=False keeps nothing server-side, so the model's reasoning only
        # carries into the next request if it comes back encrypted and is sent
        # again (messages_to_responses_input).
        kwargs["include"] = [_REASONING_INCLUDE]
    if tools:
        kwargs["tools"] = _to_responses_tools(tools)
    elif text_format is not None:
        kwargs["text"] = text_format
    return kwargs


def _mentions_json(input_items: list[dict]) -> bool:
    """Whether any input item's text contains "json", as JSON mode requires."""
    return "json" in _json.dumps(
        [item.get("content") for item in input_items if "content" in item]
    ).lower()


def _keeps_reasoning(config: LLMConfig) -> bool:
    """Whether to ask for the model's encrypted reasoning.

    Models the registry knows not to reason are left out: they have no
    reasoning to return. A model it doesn't know (often a gateway route) is
    asked, since every current OpenAI model reasons.
    """
    registered = get_model(config.provider, config.model_name)
    return registered is None or registered.uses_max_completion_tokens


def _responses_reasoning_items(response) -> list[dict] | None:
    """The response's reasoning items, as input items for the next request."""
    items: list[dict] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        raw = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        # The input item needs id, type and summary (empty is fine) and carries
        # the encrypted content back; nothing else of the output is replayed.
        payload: dict = {
            "type": "reasoning",
            "id": raw.get("id") or "",
            "summary": [
                {"type": s.get("type") or "summary_text", "text": s.get("text") or ""}
                for s in raw.get("summary") or []
                if isinstance(s, dict)
            ],
        }
        if raw.get("encrypted_content"):
            payload["encrypted_content"] = raw["encrypted_content"]
        items.append(payload)
    return items or None


def assistant_tool_turn(response: LLMResponse) -> dict:
    """The history message for a response that called tools.

    Carries the response's reasoning, which ``messages_to_responses_input``
    sends back with the next request.
    """
    message: dict = {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function_name, "arguments": _json.dumps(tc.arguments)},
            }
            for tc in response.tool_calls or []
        ],
    }
    if response.reasoning_items:
        message["reasoning_items"] = response.reasoning_items
    return message


def _responses_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if value:
                parts.append(value)
    return "".join(parts)


def _parse_tool_arguments(raw: object) -> tuple[dict, str | None]:
    """Parse a tool call's arguments without raising.

    Returns ``(arguments, parse_error)``. Models sometimes send truncated or
    badly escaped JSON, or an empty string for no-argument tools. Raising here
    would discard the whole exploration, so the error is handed back for the
    agent loop to report to the model instead.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except _json.JSONDecodeError as e:
            return {}, f"arguments were not valid JSON ({e})"
    if not isinstance(raw, dict):
        return {}, f"arguments must be a JSON object, not {type(raw).__name__}"
    return raw, None


def _responses_tool_calls(response) -> list[ToolCallResult] | None:
    results: list[ToolCallResult] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        arguments, parse_error = _parse_tool_arguments(getattr(item, "arguments", None))
        results.append(
            ToolCallResult(
                id=getattr(item, "call_id", None) or getattr(item, "id", "") or "",
                function_name=getattr(item, "name", "") or "",
                arguments=arguments,
                parse_error=parse_error,
            )
        )
    return results or None


def _extract_usage(response) -> dict[str, int | None]:
    """Pull token usage off a Responses API response, as ``LLMResponse`` fields.

    Returns ``prompt_tokens``, ``completion_tokens``, ``cached_prompt_tokens``
    and ``uncached_prompt_tokens``; any not reported are ``None``. A missing or
    all-zero usage reads as unknown, never zero: a completion that happened
    consumed prompt tokens, and a caller that took "not reported" for zero would
    think a token budget had nothing spent and run unbounded.
    """
    unknown: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_prompt_tokens": None,
        "uncached_prompt_tokens": None,
    }
    usage = getattr(response, "usage", None)
    if usage is None:
        return unknown
    prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "output_tokens", None)
    if not prompt_tokens and not completion_tokens:
        return unknown
    cached, uncached = _extract_cache_split(usage, prompt_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": uncached,
    }


def _extract_cache_split(usage, prompt_tokens: int | None) -> tuple[int | None, int | None]:
    """Split ``prompt_tokens`` into ``(cached, uncached)`` prompt-cache tokens.

    The uncached side is derived from ``prompt_tokens`` so the two always
    reconcile with it. Returns ``(None, None)`` when nothing was reported about
    caching.
    """
    details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    if not isinstance(cached, int) or not isinstance(prompt_tokens, int):
        return None, None
    cached = min(cached, prompt_tokens)
    return cached, prompt_tokens - cached


def _responses_thinking(response) -> str | None:
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        for block in getattr(item, "summary", None) or []:
            value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if value:
                parts.append(value)
    return "\n\n".join(parts) if parts else None


def _call_responses(
    config: LLMConfig,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Call OpenAI via the Responses API."""
    client = _openai_client(config)
    kwargs = _responses_call_kwargs(config, messages, tools)
    response = _stream_response(client, kwargs, call_timeout(config))

    return LLMResponse(
        content=_responses_text(response),
        thinking=_responses_thinking(response),
        model=config.model_name,
        tool_calls=_responses_tool_calls(response) if tools else None,
        reasoning_items=_responses_reasoning_items(response),
        **_extract_usage(response),
    )


def _stream_died(error: openai.APIError, *, opened: bool) -> bool:
    """Whether ``error`` lost a stream rather than refused the request.

    A refusal (a bad request, a rate limit, a context overflow) would fail the
    same way again. A connection error before the stream opened has already
    been retried by the SDK, so retrying it here would only multiply attempts.
    """
    if isinstance(error, openai.APIConnectionError):  # also APITimeoutError
        return opened
    return getattr(error, "code", None) in _DEAD_STREAM_CODES


def _event_error(event, request) -> openai.APIError | None:
    """The error an ``error`` or ``response.failed`` event reports, if any.

    The SDK raises only for a top-level ``error`` key; these carry theirs
    elsewhere and arrive as ordinary events.
    """
    kind = getattr(event, "type", None)
    if kind == "error":
        body = {"code": getattr(event, "code", None), "message": getattr(event, "message", "")}
    elif kind == "response.failed":
        error = getattr(getattr(event, "response", None), "error", None)
        body = {
            "code": getattr(error, "code", None),
            "message": getattr(error, "message", None) or "The response failed.",
        }
    else:
        return None
    return openai.APIError(body["message"] or "The model stream failed.", request, body=body)


def _read_stream(client: OpenAI, kwargs: dict, limit: float, opened: list[bool]):
    """One streamed call: its final Response, or the error that ended it.

    Sets ``opened[0]`` once the response has started.
    """
    started = time.monotonic()
    stream = client.responses.create(**kwargs, stream=True)
    opened[0] = True
    request = stream.response.request
    with stream:
        for event in stream:
            if time.monotonic() - started > limit:
                # Events kept coming, so no read ever timed out.
                raise openai.APITimeoutError(request)
            error = _event_error(event, request)
            if error is not None:
                raise error
            if getattr(event, "type", None) in ("response.completed", "response.incomplete"):
                return event.response
    raise openai.APIError(
        "The model stream ended before the response completed.",
        request,
        body={"code": "server_error", "message": "stream ended before response.completed"},
    )


def _stream_response(client: OpenAI, kwargs: dict, limit: float):
    """Stream one Responses call and return the final Response.

    Every call streams, so a gateway that times out waiting for response
    headers never mistakes a slow answer for a silent one. ``limit`` bounds
    the whole stream as well as each silent read: a stream can keep sending
    events for as long as it likes, and a read timeout alone never ends it.

    A stream that dies (see ``_stream_died``) is retried up to STREAM_ATTEMPTS
    times in all, with a short backoff; any other error is raised at once.
    """
    for attempt in range(1, STREAM_ATTEMPTS + 1):
        started = time.monotonic()
        opened = [False]
        try:
            return _read_stream(client, kwargs, limit, opened)
        except openai.APIError as error:
            if attempt == STREAM_ATTEMPTS or not _stream_died(error, opened=opened[0]):
                raise
            logger.warning(
                "Model stream died after %.0fs (attempt %d of %d), retrying: %s",
                time.monotonic() - started,
                attempt,
                STREAM_ATTEMPTS,
                str(error)[:300],
            )
            _sleep(2**attempt)
    raise AssertionError("unreachable")


def _image_messages(
    config: LLMConfig,
    prompt: str,
    base64_image: str,
    media_type: str,
) -> list[dict]:
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{base64_image}"},
                },
            ],
        }
    ]
    if model_uses_completion_tokens(config.model_name):
        from stride_gpt.core.prompts import create_reasoning_system_prompt

        system_prompt = create_reasoning_system_prompt(
            task_description="Analyze the provided architecture diagram and explain it to a Security Architect.",
            approach_description="""1. Carefully examine the diagram
2. Identify all components and their relationships
3. Note any technologies, protocols, or security measures shown
4. Create a clear, structured explanation with these sections:
   - Overall Architecture: Brief overview of the system
   - Key Components: List and explain each major component
   - Data Flow: How information moves through the system
   - Technologies Used: Identify technologies, frameworks, or platforms
   - Security Considerations: Note any visible security measures""",
        )
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages

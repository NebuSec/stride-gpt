"""The Anthropic Messages API side of core/llm.py.

``core/llm.py`` routes a provider whose registry entry says ``api="messages"``
here. Agent history stays in Chat Completions shape, as it does for the
Responses path, and is converted to Messages content blocks at this boundary
(``messages_to_anthropic``).

Thinking blocks and the history-editing check
---------------------------------------------
Claude's thinking blocks are passed back unchanged with the assistant turn that
produced them, so a tool loop keeps its reasoning. Current Claude models bind
each block to the request that produced it: if the model, the ``system``
prompt, the ``tools`` array or any earlier message has changed since, the API
refuses the replayed block (a 400) for accounts that enforce that check. The
agent loop does change all three: context compression rewrites old turns into
a summary and the grace round swaps the tool set. So every assistant turn
records a hash of the request it answered (``binding``), and a turn whose
request no longer hashes the same is replayed without its thinking, rebuilt
from its text and tool calls. That is what the API's own ``drop_block`` mode
does, without the beta header a gateway might not pass on.

Never construct ``Anthropic()`` outside this module.
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import os
import time

import anthropic
import httpx2
from anthropic import Anthropic

from stride_gpt.core import llm as _llm
from stride_gpt.core.schemas import LLMConfig, LLMResponse, ToolCallResult
from stride_gpt.models import get_model

logger = logging.getLogger(__name__)

# The Messages API requires max_tokens. A model the registry doesn't know (a
# gateway route, a newer model) gets this cap; every current Claude model
# allows at least this much output, and calls stream, so no HTTP timeout bites.
DEFAULT_MAX_TOKENS = 64000
# The effort levels the Messages API accepts (output_config.effort).
_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
# Server-side refusal fallbacks (beta): on a safety refusal the API re-runs the
# same request on a fallback model it picks by refusal category.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Error types a Messages stream reports after it started (an SSE `error`
# event): the request was fine, the stream was lost.
_DEAD_STREAM_TYPES = frozenset({"overloaded_error", "api_error", "timeout_error"})
# What the SDK's stream raises when the model's tool-input JSON can't be
# parsed: a sampling fault, not a refused request, so the call is sampled again.
_BAD_TOOL_JSON = "Unable to parse tool parameter JSON"
# The reasoning_items envelope of one Claude assistant turn.
_TURN_ITEM = "anthropic_turn"
# Which fields of each response block go back as input; the response carries
# more (citations, caller, ...) that an input block does not take.
_REPLAY_FIELDS = {
    "text": ("type", "text"),
    "thinking": ("type", "thinking", "signature"),
    "redacted_thinking": ("type", "data"),
    "tool_use": ("type", "id", "name", "input"),
}


def call_messages(
    config: LLMConfig,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Call Claude via the Messages API (streamed)."""
    kwargs = messages_call_kwargs(config, messages, tools)
    binding = request_binding(kwargs)
    with _anthropic_client(config) as client:
        stream_fn = client.beta.messages.stream if "fallbacks" in kwargs else client.messages.stream
        message = _stream_message(stream_fn, kwargs, _llm.call_timeout(config))

    answer = _answer_blocks(message, structured="format" in kwargs.get("output_config", {}))
    content = "".join(
        getattr(block, "text", "") or "" for block in answer if getattr(block, "type", None) == "text"
    )
    stop_reason = getattr(message, "stop_reason", None)
    # Both stops end generation wherever it was, possibly inside a tool call.
    cut_off = stop_reason in ("max_tokens", "model_context_window_exceeded")
    tool_calls = _tool_calls(_after_last_fallback(message), truncated=cut_off) if tools else None

    if stop_reason == "refusal":
        # The fallback chain, if any, refused too. Partial output after a
        # refusal is not an answer.
        raise _llm.ModelRefusalError(_refusal_text(message))
    incomplete_reason = _incomplete_reason(stop_reason)
    if incomplete_reason is not None:
        logger.warning(
            "Model response is incomplete (%s); its content and tool calls are partial.",
            incomplete_reason,
        )

    return LLMResponse(
        content=content,
        thinking=_thinking_text(message),
        model=getattr(message, "model", None) or config.model_name,
        tool_calls=tool_calls,
        reasoning_items=[
            {
                "type": _TURN_ITEM,
                # A turn a fallback model served carries that model's thinking,
                # bound to its own request, and a turn with a call whose input
                # is no JSON object can't go back as the API takes a tool_use:
                # both are rebuilt from their text and parsed tool calls.
                "binding": None
                if _fallback_served(message)
                or _has_unreplayable_call(message)
                or _has_unreplayable_thinking(message)
                else binding,
                "content": _replay_content(message),
            }
        ],
        incomplete_reason=incomplete_reason,
        **_extract_usage(message),
    )


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def _anthropic_client(config: LLMConfig) -> Anthropic:
    # Same api_key / base_url / retries / timeout rules as the OpenAI client.
    # The SDK appends /v1/messages to base_url, so a custom base names the
    # host (or gateway prefix), not .../v1.
    kwargs = _llm._client_kwargs(config)
    # A bare number would bound the connect as well as the read: an
    # unreachable gateway would hang for the whole call limit per attempt.
    kwargs["timeout"] = httpx2.Timeout(kwargs["timeout"], connect=min(10.0, kwargs["timeout"]))
    return Anthropic(**kwargs)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def messages_call_kwargs(
    config: LLMConfig,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """Kwargs for client.messages.stream()."""
    anthropic_tools = to_anthropic_tools(tools) if tools else None
    system, converted = messages_to_anthropic(
        messages, model=config.model_name, tools=anthropic_tools
    )
    registered = get_model(config.provider, config.model_name)
    kwargs: dict = {
        "model": config.model_name,
        "max_tokens": config.max_tokens
        or (registered.default_tokens if registered is not None else DEFAULT_MAX_TOKENS),
        "messages": converted,
        # Automatic prompt caching: the agent loop resends a growing history
        # on every turn, so each call reads the previous one's prefix.
        "cache_control": {"type": "ephemeral"},
    }
    if system:
        kwargs["system"] = system
    if anthropic_tools:
        kwargs["tools"] = anthropic_tools
    output_config: dict = {}
    if config.reasoning_effort:
        problem = effort_problem(config.reasoning_effort)
        if problem:
            raise ValueError(problem)
        output_config["effort"] = config.reasoning_effort
    if isinstance(config.response_format, dict) and not tools:
        # Structured outputs need additionalProperties: false on every object
        # and drop constraints the API can't enforce; transform_schema does
        # both. "json" has no Messages equivalent: those prompts ask for JSON
        # themselves and their callers extract it from the text.
        output_config["format"] = {
            "type": "json_schema",
            "schema": anthropic.transform_schema(config.response_format),
        }
    if output_config:
        kwargs["output_config"] = output_config
    if registered is not None and registered.refusal_fallback and not _gateway(config):
        # Only against Anthropic itself: a gateway routes models its own way
        # and need not pass on a beta header or field it doesn't know, and a
        # request it refuses would fail every call.
        kwargs["betas"] = [_FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    return kwargs


def effort_problem(effort: str) -> str | None:
    """Why the Messages API would refuse ``effort``, or None if it takes it."""
    if effort in _EFFORTS:
        return None
    return (
        f"Reasoning effort {effort!r} is not one the Messages API takes; "
        f"use one of {', '.join(sorted(_EFFORTS))}."
    )


def _gateway(config: LLMConfig) -> bool:
    """Whether calls go somewhere other than Anthropic's own API."""
    # The SDK reads ANTHROPIC_BASE_URL itself when no base_url is passed.
    return bool(config.api_base or os.environ.get("ANTHROPIC_BASE_URL"))


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Chat Completions nested function tools → Messages tools."""
    converted: list[dict] = []
    for tool in tools:
        fn = tool.get("function") if tool.get("type") == "function" else None
        if not isinstance(fn, dict):
            converted.append(tool)
            continue
        converted.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


def request_binding(kwargs: dict) -> str:
    """Hash of what a response's thinking is bound to: model, system, tools, messages."""
    return _binding(kwargs["model"], kwargs.get("system"), kwargs.get("tools"), kwargs["messages"])


def _binding(model: str, system: str | None, tools: list[dict] | None, prior: list[dict]) -> str:
    payload = _json.dumps(
        {"model": model, "system": system, "tools": tools, "messages": prior},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def messages_to_anthropic(
    messages: list[dict],
    *,
    model: str,
    tools: list[dict] | None = None,
) -> tuple[str | None, list[dict]]:
    """Split Chat Completions-shaped messages into (system, Messages messages).

    The system messages that open the conversation become ``system``; a later
    one stays where it was said, as a mid-conversation system message. Tool
    results become ``tool_result`` blocks, joined into one user message per
    round as the API wants parallel results. An assistant turn is replayed
    exactly as the model produced it, thinking included, while the request it
    answered still hashes the same (see the module docstring); otherwise it is
    rebuilt from its text and tool calls.
    """
    system_parts: list[str] = []
    out: list[dict] = []
    leading = True
    system: str | None = None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system" and leading and isinstance(content, str):
            if content:
                system_parts.append(content)
            continue
        if leading:
            leading = False
            system = "\n\n".join(system_parts) or None

        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id") or "",
                "content": content if isinstance(content, str) else _json.dumps(content),
            }
            last = out[-1] if out else None
            if last is not None and last["role"] == "user" and _is_tool_results(last):
                last["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue

        if role == "assistant":
            replay = _bound_turn(msg, model=model, system=system, tools=tools, prior=out)
            if replay is not None:
                out.append({"role": "assistant", "content": replay})
                continue
            blocks: list[dict] = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                arguments, _ = _llm._parse_tool_arguments(fn.get("arguments"))
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or tc.get("call_id") or "",
                        "name": fn.get("name") or "",
                        "input": arguments,
                    }
                )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
            continue

        converted = _convert_content_parts(content)
        last = out[-1] if out else None
        if (role or "user") == "user" and last is not None and last["role"] == "user":
            # One user turn after another (a nudge after tool results): joined
            # into one, since a strict Messages gateway insists turns alternate.
            last["content"] = [*_as_blocks(last["content"]), *_as_blocks(converted)]
            continue
        out.append({"role": role or "user", "content": converted})

    if leading:
        system = "\n\n".join(system_parts) or None
    if not out and system:
        # The API refuses an empty messages list: a system-only request is
        # sent as the user message instead.
        out = [{"role": "user", "content": system}]
        system = None
    return system, out


def _as_blocks(content: object) -> list:
    if isinstance(content, list):
        return list(content)
    return [{"type": "text", "text": content}] if content else []


def _is_tool_results(message: dict) -> bool:
    content = message.get("content")
    return (
        isinstance(content, list)
        and bool(content)
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    )


def _bound_turn(
    msg: dict,
    *,
    model: str,
    system: str | None,
    tools: list[dict] | None,
    prior: list[dict],
) -> list[dict] | None:
    """The assistant turn's own content blocks, if its request is unchanged."""
    for item in msg.get("reasoning_items") or []:
        if not isinstance(item, dict) or item.get("type") != _TURN_ITEM:
            continue
        content = item.get("content")
        if content and item.get("binding") == _binding(model, system, tools, prior):
            return [dict(block) for block in content]
        return None
    return None


def _convert_content_parts(content: object) -> object:
    """Chat Completions content parts → Messages text / image blocks."""
    if not isinstance(content, list):
        return content
    converted: list = []
    for part in content:
        if not isinstance(part, dict):
            converted.append(part)
            continue
        ptype = part.get("type")
        if ptype == "text":
            converted.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            converted.append({"type": "image", "source": _image_source(url or "")})
        else:
            converted.append(part)
    return converted


def _image_source(url: str) -> dict:
    """A data: URL becomes a base64 source; anything else is fetched by URL."""
    if url.startswith("data:") and ";base64," in url:
        header, data = url[len("data:"):].split(";base64,", 1)
        return {"type": "base64", "media_type": header or "image/jpeg", "data": data}
    return {"type": "url", "url": url}


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def _replay_content(message) -> list[dict]:
    """The response's content blocks as input blocks, in the order produced."""
    blocks: list[dict] = []
    for block in getattr(message, "content", None) or []:
        # Read field by field: a dump would also carry response-only fields,
        # and warns on an input the model sent as something other than an object.
        kind = getattr(block, "type", None)
        raw = {key: getattr(block, key) for key in _REPLAY_FIELDS.get(kind, ()) if hasattr(block, key)}
        fields = _REPLAY_FIELDS.get(kind)
        if fields is None:
            # Fallback markers and other response-only blocks are not input.
            continue
        if raw.get("type") == "text" and not raw.get("text"):
            continue  # the API refuses an empty text block
        blocks.append({key: raw[key] for key in fields if key in raw})
    return blocks


def _tool_calls(message, *, truncated: bool = False) -> list[ToolCallResult] | None:
    """The message's tool calls.

    When the output limit stopped the message, its last block may be a call
    whose input the stream closed early: the SDK completes partial JSON, so
    those arguments parse but are not what the model meant. That call is
    handed back as unparseable instead of run.
    """
    results: list[ToolCallResult] = []
    blocks = list(getattr(message, "content", None) or [])
    for position, block in enumerate(blocks):
        if getattr(block, "type", None) != "tool_use":
            continue
        raw = getattr(block, "input", None)
        arguments, parse_error = _llm._parse_tool_arguments(raw)
        if truncated and position == len(blocks) - 1:
            arguments, parse_error = {}, "arguments were cut off at the output token limit"
        results.append(
            ToolCallResult(
                id=getattr(block, "id", "") or "",
                function_name=getattr(block, "name", "") or "",
                arguments=arguments,
                parse_error=parse_error,
                raw_arguments=raw if isinstance(raw, str) else None,
            )
        )
    return results or None


def _after_last_fallback(message):
    """The message with only the blocks after its last fallback marker.

    On a mid-stream fallback the content holds the declined model's blocks,
    a ``fallback`` marker, then the fallback model's. A call the declined
    model made was never seen by the model that finished the turn, and is
    not to be run or answered.
    """
    blocks = list(getattr(message, "content", None) or [])
    seams = [i for i, b in enumerate(blocks) if getattr(b, "type", None) == "fallback"]
    if not seams:
        return message
    return message.model_copy(update={"content": blocks[seams[-1] + 1 :]})


def _answer_blocks(message, *, structured: bool) -> list:
    """The blocks whose text is the answer.

    A fallback that continues the declined model's text keeps it; one that
    starts over (always, for a structured output) makes it a false start that
    would corrupt the JSON.
    """
    if structured:
        return list(getattr(_after_last_fallback(message), "content", None) or [])
    return list(getattr(message, "content", None) or [])


def _has_unreplayable_thinking(message) -> bool:
    """Thinking the API would refuse back: unsigned, or ending the turn."""
    blocks = list(getattr(message, "content", None) or [])
    for block in blocks:
        kind = getattr(block, "type", None)
        if kind == "thinking" and not getattr(block, "signature", None):
            return True
        if kind == "redacted_thinking" and not getattr(block, "data", None):
            return True
    # Judged on what would be replayed: empty text blocks are dropped there.
    replayed = _replay_content(message)
    return bool(replayed) and replayed[-1]["type"] in ("thinking", "redacted_thinking")


def _has_unreplayable_call(message) -> bool:
    return any(
        getattr(block, "type", None) == "tool_use"
        and not isinstance(getattr(block, "input", None), dict)
        for block in getattr(message, "content", None) or []
    )


def _fallback_served(message) -> bool:
    """Whether a server-side fallback model produced (part of) this message."""
    if any(getattr(b, "type", None) == "fallback" for b in getattr(message, "content", None) or []):
        return True
    iterations = getattr(getattr(message, "usage", None), "iterations", None) or []
    return any(getattr(i, "type", None) == "fallback_message" for i in iterations)


def _thinking_text(message) -> str | None:
    parts = [
        block.thinking
        for block in getattr(message, "content", None) or []
        if getattr(block, "type", None) == "thinking" and getattr(block, "thinking", None)
    ]
    return "\n\n".join(parts) if parts else None


def _refusal_text(message) -> str:
    details = getattr(message, "stop_details", None)
    explanation = getattr(details, "explanation", None)
    category = getattr(details, "category", None)
    if explanation:
        return explanation
    if category:
        return f"The model declined to answer (category: {category})."
    return "The model declined to answer."


def _incomplete_reason(stop_reason: str | None) -> str | None:
    """Why the message stopped early, in the Responses path's vocabulary."""
    if stop_reason in (None, "end_turn", "tool_use", "stop_sequence"):
        return None
    if stop_reason == "max_tokens":
        return "max_output_tokens"
    return stop_reason  # model_context_window_exceeded, pause_turn, ...


def _extract_usage(message) -> dict[str, int | None]:
    """Token usage as ``LLMResponse`` fields; see ``core/llm.py:_extract_usage``.

    ``input_tokens`` counts only what came after the last cache breakpoint, so
    the prompt total adds the cache reads and writes back.
    """
    unknown: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_prompt_tokens": None,
        "uncached_prompt_tokens": None,
    }
    usage = getattr(message, "usage", None)
    if usage is None:
        return unknown
    read_reported = getattr(usage, "cache_read_input_tokens", None)
    written_reported = getattr(usage, "cache_creation_input_tokens", None)
    read = read_reported or 0
    fresh = getattr(usage, "input_tokens", None) or 0
    completion = getattr(usage, "output_tokens", None)
    prompt = fresh + read + (written_reported or 0)
    if not prompt and not completion:
        return unknown
    # Neither cache field reported means nothing was said about caching
    # (a gateway that strips them), not that nothing was cached.
    cache_reported = bool(prompt) and (read_reported is not None or written_reported is not None)
    return {
        "prompt_tokens": prompt or None,
        "completion_tokens": completion,
        "cached_prompt_tokens": read if cache_reported else None,
        "uncached_prompt_tokens": prompt - read if cache_reported else None,
    }


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _error_type(error: anthropic.APIError) -> str | None:
    """The Messages error type (``overloaded_error``, ...) an error carries."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and isinstance(inner.get("type"), str):
            return inner["type"]
        if isinstance(body.get("type"), str) and body["type"] != "error":
            return body["type"]
    return None


def _stream_died(error: anthropic.APIError, *, opened: bool) -> bool:
    """Whether ``error`` lost a started stream rather than refused the request.

    Before the stream opened, the SDK has already retried what is worth
    retrying (connection errors, 408, 429, 5xx, 529).
    """
    if not opened:
        return False
    if isinstance(error, anthropic.APIConnectionError):  # also APITimeoutError
        return True
    return _error_type(error) in _DEAD_STREAM_TYPES


def _read_stream(stream_fn, kwargs: dict, limit: float, opened: list[bool]):
    """One streamed call: its final Message, or the error that ended it."""
    with stream_fn(**kwargs) as stream:
        # From the open: the SDK's own retries before it (their backoff and
        # Retry-After) are not part of this stream's time.
        started = time.monotonic()
        opened[0] = True
        request = stream.response.request
        try:
            for _event in stream:
                if time.monotonic() - started > limit:
                    # Events kept coming, so no read ever timed out.
                    raise anthropic.APITimeoutError(request)
        except httpx2.TimeoutException as error:
            # The SDK wraps transport errors only before the response starts;
            # once the body is streaming they arrive raw.
            raise anthropic.APITimeoutError(request) from error
        except httpx2.TransportError as error:
            raise anthropic.APIConnectionError(
                message=f"The model stream was lost: {error}", request=request
            ) from error
        except (httpx2.DecodingError, UnicodeDecodeError) as error:
            # A body that can't be decoded (corrupt gzip, a broken UTF-8 byte)
            # arrives raw too: the stream was damaged, not refused.
            raise anthropic.APIConnectionError(
                message=f"The model stream could not be decoded: {error}", request=request
            ) from error
        except _json.JSONDecodeError as error:
            # An event whose data is not JSON: the stream was cut or garbled
            # on the way (a gateway), not refused.
            raise anthropic.APIConnectionError(
                message="The model stream sent an unreadable event.", request=request
            ) from error
        except ValueError as error:
            if _BAD_TOOL_JSON not in str(error):
                raise
            raise anthropic.APIConnectionError(
                message="The model's tool-call JSON could not be parsed.", request=request
            ) from error
        try:
            message = stream.get_final_message()
        except AssertionError:  # the stream ended before message_start
            message = None
    if message is None or getattr(message, "stop_reason", None) is None:
        raise anthropic.APIConnectionError(
            message="The model stream ended before the message completed.", request=request
        )
    return message


def _stream_message(stream_fn, kwargs: dict, limit: float):
    """Stream one Messages call and return the final Message.

    Same contract as ``core/llm.py:_stream_response``: ``limit`` bounds the
    whole stream, and a stream that dies after it started is retried up to
    STREAM_ATTEMPTS times in all, with a short backoff.
    """
    for attempt in range(1, _llm.STREAM_ATTEMPTS + 1):
        started = time.monotonic()
        opened = [False]
        try:
            return _read_stream(stream_fn, kwargs, limit, opened)
        except anthropic.APIError as error:
            if attempt == _llm.STREAM_ATTEMPTS or not _stream_died(error, opened=opened[0]):
                raise
            logger.warning(
                "Model stream died after %.0fs (attempt %d of %d), retrying: %s",
                time.monotonic() - started,
                attempt,
                _llm.STREAM_ATTEMPTS,
                str(error)[:300],
            )
            _llm._sleep(2**attempt)
    raise AssertionError("unreachable")

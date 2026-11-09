"""Unified LLM interface: the OpenAI SDK against the Responses API.

Agent history stays in Chat Completions shape and is converted at this boundary.
"""

from __future__ import annotations

import json as _json

from openai import OpenAI

from stride_gpt.core.schemas import LLMConfig, LLMResponse, ToolCallResult
from stride_gpt.models import get_model, get_provider_by_key, model_uses_completion_tokens


def _default_output_tokens(config: LLMConfig, fallback: int) -> int:
    """Resolve the output-token cap: caller override > registry default > fallback."""
    if config.max_tokens:
        return config.max_tokens
    registered = get_model(config.provider, config.model_name)
    if registered is not None:
        return registered.default_tokens
    return fallback


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
    }
    if config.timeout:
        kwargs["timeout"] = config.timeout
    if config.api_base:
        kwargs["base_url"] = config.api_base.rstrip("/")
    return kwargs


def _openai_client(config: LLMConfig) -> OpenAI:
    return OpenAI(**_client_kwargs(config))


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


def _responses_text_format(config: LLMConfig) -> dict | None:
    """Responses API `text.format` payload, or None to omit."""
    if isinstance(config.response_format, dict):
        return {
            "format": {
                "type": "json_schema",
                "name": "response",
                "schema": config.response_format,
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
            converted.append({"type": "input_image", "image_url": url})
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

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str) and content:
                instructions_parts.append(content)
            continue

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
    return instructions, items


def _responses_call_kwargs(
    config: LLMConfig,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """Kwargs for client.responses.create()."""
    instructions, input_items = messages_to_responses_input(messages)
    kwargs: dict = {
        "model": config.model_name,
        "input": input_items,
        "store": False,
        "max_output_tokens": _default_output_tokens(config, 20000),
    }
    if instructions:
        kwargs["instructions"] = instructions
    if tools:
        kwargs["tools"] = _to_responses_tools(tools)
    else:
        text_format = _responses_text_format(config)
        if text_format is not None:
            kwargs["text"] = text_format
    return kwargs


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
    response = client.responses.create(**kwargs)

    return LLMResponse(
        content=_responses_text(response),
        thinking=_responses_thinking(response),
        model=config.model_name,
        tool_calls=_responses_tool_calls(response) if tools else None,
        **_extract_usage(response),
    )


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

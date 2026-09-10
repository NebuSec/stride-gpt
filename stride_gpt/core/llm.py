"""Unified LLM interface.

OpenAI uses the native Responses API. Groq, DeepSeek, Mistral, and LM Studio
use the OpenAI SDK against Chat Completions-compatible endpoints.
"""

from __future__ import annotations

import json as _json
import re

from openai import OpenAI

from stride_gpt.core.mermaid_utils import extract_mermaid_code
from stride_gpt.core.schemas import LLMConfig, LLMResponse, ToolCallResult
from stride_gpt.models import (
    get_model,
    get_provider_by_key,
    model_uses_completion_tokens,
    provider_uses_responses,
)

# Local OpenAI-compatible servers (LM Studio) ignore the key, but the OpenAI SDK
# refuses to construct a client without one. Matches apps/web's convention.
LOCAL_PLACEHOLDER_API_KEY = "not-needed"


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
    if provider_uses_responses(config.provider):
        return _call_responses(config, messages)
    return _call_chat_completions(config, messages)


def call_llm_with_tools(
    config: LLMConfig, messages: list[dict], tools: list[dict]
) -> LLMResponse:
    """Call an LLM with tool definitions. Returns response with optional tool_calls."""
    if provider_uses_responses(config.provider):
        return _call_responses(config, messages, tools=tools)
    return _call_chat_completions(config, messages, tools=tools)


def call_llm_with_image(
    config: LLMConfig,
    prompt: str,
    base64_image: str,
    media_type: str = "image/jpeg",
) -> LLMResponse:
    """Call an LLM with an image for analysis."""
    messages = _image_messages(config, prompt, base64_image, media_type)
    if provider_uses_responses(config.provider):
        return _call_responses(config, messages)
    return _call_chat_completions(config, messages)


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def _client_kwargs(config: LLMConfig) -> dict:
    """Kwargs for OpenAI() — api_key, base_url, retries, timeout."""
    provider_info = get_provider_by_key(config.provider)
    api_key = config.api_key or None
    if api_key is None and provider_info is not None and not provider_info.needs_api_key:
        api_key = LOCAL_PLACEHOLDER_API_KEY

    kwargs: dict = {
        "api_key": api_key,
        "max_retries": 3,
    }
    if config.timeout:
        kwargs["timeout"] = config.timeout

    base_url = _base_url(config)
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def _base_url(config: LLMConfig) -> str | None:
    """Resolve the OpenAI-SDK base_url, or None for the default OpenAI host."""
    if config.api_base:
        base = config.api_base.rstrip("/")
        if config.provider == "LM Studio Server" and not base.endswith("/v1"):
            return f"{base}/v1"
        return base
    provider_info = get_provider_by_key(config.provider)
    if provider_info is not None and provider_info.chat_api_base:
        return provider_info.chat_api_base
    return None


def _openai_client(config: LLMConfig) -> OpenAI:
    return OpenAI(**_client_kwargs(config))


# ---------------------------------------------------------------------------
# JSON / token kwargs shared by both transports
# ---------------------------------------------------------------------------


def _json_response_format(config: LLMConfig) -> dict | None:
    """Chat Completions `response_format` payload, or None to omit."""
    if isinstance(config.response_format, dict):
        return {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": config.response_format},
        }
    if config.response_format == "json" and config.provider != "LM Studio Server":
        # LM Studio only accepts json_schema or text.
        return {"type": "json_object"}
    return None


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


def _chat_token_kwargs(config: LLMConfig) -> dict:
    """max_tokens / max_completion_tokens for Chat Completions."""
    if model_uses_completion_tokens(config.model_name):
        return {"max_completion_tokens": _default_output_tokens(config, 20000)}
    if config.provider == "LM Studio Server":
        return {"max_tokens": config.max_tokens or 8000}
    # Groq / Mistral / DeepSeek: only set if the caller asked.
    if config.max_tokens:
        return {"max_tokens": config.max_tokens}
    return {}


# ---------------------------------------------------------------------------
# Responses API (OpenAI)
# ---------------------------------------------------------------------------


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


def _responses_tool_calls(response) -> list[ToolCallResult] | None:
    results: list[ToolCallResult] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        raw_args = getattr(item, "arguments", None) or "{}"
        parsed = _json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        results.append(
            ToolCallResult(
                id=getattr(item, "call_id", None) or getattr(item, "id", "") or "",
                function_name=getattr(item, "name", "") or "",
                arguments=parsed if isinstance(parsed, dict) else {},
            )
        )
    return results or None


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

    content = _responses_text(response)
    tool_calls = _responses_tool_calls(response) if tools else None
    thinking = _responses_thinking(response)

    return LLMResponse(
        content=content,
        thinking=thinking,
        model=config.model_name,
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# Chat Completions (OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------


def _chat_call_kwargs(
    config: LLMConfig,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    kwargs: dict = {
        "model": config.model_name,
        "messages": messages,
        **_chat_token_kwargs(config),
    }
    if tools:
        kwargs["tools"] = tools
    else:
        response_format = _json_response_format(config)
        if response_format is not None:
            kwargs["response_format"] = response_format
    return kwargs


def _call_chat_completions(
    config: LLMConfig,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Call an OpenAI-compatible Chat Completions endpoint."""
    client = _openai_client(config)
    kwargs = _chat_call_kwargs(config, messages, tools)
    response = client.chat.completions.create(**kwargs)

    message = response.choices[0].message
    content = message.content or ""

    # Some local reasoning models emit the structured output into
    # `reasoning_content` and leave `content` empty.
    if not content.strip():
        fallback = getattr(message, "reasoning_content", None)
        if fallback:
            content = fallback

    tool_calls = None
    if tools and getattr(message, "tool_calls", None):
        tool_calls = [
            ToolCallResult(
                id=tc.id,
                function_name=tc.function.name,
                arguments=_json.loads(tc.function.arguments)
                if isinstance(tc.function.arguments, str)
                else tc.function.arguments,
            )
            for tc in message.tool_calls
        ]

    reasoning = None
    if config.provider == "Groq API":
        reasoning, content = extract_deepseek_reasoning(content)

    return LLMResponse(
        content=content,
        reasoning=reasoning,
        model=config.model_name,
        tool_calls=tool_calls,
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


# ---------------------------------------------------------------------------
# Utility functions (moved from utils.py)
# ---------------------------------------------------------------------------


def extract_deepseek_reasoning(response_text: str) -> tuple[str | None, str]:
    """Extract reasoning from <think></think> tags in DeepSeek R1 responses.

    Returns:
        Tuple of (reasoning, final_output). Reasoning is None if no tags found.
    """
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, response_text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        final_output = re.sub(think_pattern, "", response_text, flags=re.DOTALL).strip()
        return reasoning, final_output
    return None, response_text


def process_groq_response(
    response_text: str, model_name: str, expect_json: bool = True
) -> tuple[str | None, str | dict]:
    """Process a Groq API response, handling DeepSeek R1 reasoning extraction.

    Returns:
        Tuple of (reasoning, processed_output).
    """
    reasoning = None
    final_output = response_text

    if model_name == "deepseek-r1-distill-llama-70b":
        reasoning, final_output = extract_deepseek_reasoning(response_text)

    if expect_json:
        try:
            processed_output = _json.loads(final_output)
        except _json.JSONDecodeError:
            processed_output = final_output
    elif "graph " in final_output:
        processed_output = extract_mermaid_code(final_output)
    else:
        processed_output = final_output

    return reasoning, processed_output

"""Provider-specific adapters for cloud backends.

This module provides adapters for different LLM API providers that handle:
- Authentication headers
- Request body transformation (to provider-specific format)
- Response body transformation (to OpenAI-compatible format)
- SSE stream chunk transformation
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ProviderAdapter(ABC):
    """Base class for cloud provider adapters."""

    @abstractmethod
    def get_headers(self, api_key: str, base_headers: Dict[str, str]) -> Dict[str, str]:
        """Return headers with provider-specific auth.

        Args:
            api_key: The API key for this provider
            base_headers: Existing headers to augment

        Returns:
            Headers dict with auth and content-type set
        """
        pass

    def transform_request(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Transform request body for this provider. Default: passthrough."""
        return body

    def transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform response to OpenAI format. Default: passthrough."""
        return response

    def transform_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform SSE chunk to OpenAI format. Default: passthrough.

        Returns None to skip/filter the chunk.
        """
        return chunk

    def parse_sse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse an SSE line into a chunk dict. Default: OpenAI format.

        Override for providers with different SSE formats.
        """
        if not line.startswith("data: "):
            return None
        data = line[6:]
        if data == "[DONE]":
            return {"done": True}
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    def get_chat_endpoint(self) -> str:
        """Return chat completions endpoint path."""
        return "/chat/completions"

    def get_embeddings_endpoint(self) -> str:
        """Return embeddings endpoint path."""
        return "/embeddings"


class OpenAIAdapter(ProviderAdapter):
    """OpenAI and OpenAI-compatible providers."""

    def get_headers(self, api_key: str, base_headers: Dict[str, str]) -> Dict[str, str]:
        # Filter out headers that we'll set explicitly (case-insensitive)
        skip_headers = {'content-type', 'accept', 'authorization'}
        headers = {k: v for k, v in base_headers.items() if k.lower() not in skip_headers}
        # Provider-specific base headers
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        # OpenAI auth
        headers["Authorization"] = f"Bearer {api_key}"
        return headers


class AnthropicAdapter(ProviderAdapter):
    """Anthropic Claude API adapter.

    Handles transformation between OpenAI format and Anthropic's native format.
    """

    def get_headers(self, api_key: str, base_headers: Dict[str, str]) -> Dict[str, str]:
        # Filter out headers that we'll set explicitly (case-insensitive)
        skip_headers = {'content-type', 'accept', 'authorization', 'x-api-key'}
        headers = {k: v for k, v in base_headers.items() if k.lower() not in skip_headers}
        # Provider-specific base headers
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        # Anthropic auth
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        # Enable tool streaming (required for consistent tool_calls behavior in SSE)
        headers["anthropic-beta"] = "tools-2024-04-04"
        return headers

    def transform_request(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Transform OpenAI format to Anthropic format."""
        messages = body.get("messages", [])
        system_msg = None
        chat_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                # Anthropic handles system messages separately
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_msg = content
                elif isinstance(content, list):
                    # Handle structured content
                    system_msg = " ".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
            else:
                # Transform message content for images
                content = msg.get("content")
                if isinstance(content, list):
                    # OpenAI vision format -> Anthropic format
                    anthropic_content = []
                    for part in content:
                        if part.get("type") == "text":
                            anthropic_content.append({"type": "text", "text": part["text"]})
                        elif part.get("type") == "image_url":
                            # Convert OpenAI image_url to Anthropic source format
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                # Base64 encoded
                                try:
                                    media_type, b64_data = url.split(";base64,")
                                    media_type = media_type.replace("data:", "")
                                    anthropic_content.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": b64_data
                                        }
                                    })
                                except ValueError:
                                    # Malformed data URL, skip
                                    logger.warning("Skipping malformed data URL in image_url")
                            else:
                                # URL reference (Anthropic supports URL type)
                                anthropic_content.append({
                                    "type": "image",
                                    "source": {"type": "url", "url": url}
                                })
                    chat_messages.append({"role": msg["role"], "content": anthropic_content})
                else:
                    chat_messages.append({"role": msg["role"], "content": content})

        result = {
            "model": body.get("model"),
            "messages": chat_messages,
            "max_tokens": body.get("max_tokens", 4096),
        }
        if system_msg:
            result["system"] = system_msg
        if "temperature" in body:
            result["temperature"] = body["temperature"]
        if "stream" in body:
            result["stream"] = body["stream"]
        if "top_p" in body:
            result["top_p"] = body["top_p"]
        if "stop" in body:
            result["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]

        # Transform tools (OpenAI format -> Anthropic format)
        if "tools" in body:
            anthropic_tools = []
            for tool in body["tools"]:
                if tool.get("type") == "function":
                    func = tool["function"]
                    anthropic_tools.append({
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                    })
            result["tools"] = anthropic_tools

        return result

    def transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Anthropic response to OpenAI format."""
        content_blocks = response.get("content", [])
        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                # Anthropic tool_use -> OpenAI tool_calls
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })

        message = {
            "role": "assistant",
            "content": text_content if text_content else None
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        # Map usage tokens (preserve structure)
        anthropic_usage = response.get("usage", {})
        openai_usage = {
            "prompt_tokens": anthropic_usage.get("input_tokens", 0),
            "completion_tokens": anthropic_usage.get("output_tokens", 0),
            "total_tokens": anthropic_usage.get("input_tokens", 0) + anthropic_usage.get("output_tokens", 0)
        }

        return {
            "id": response.get("id", ""),
            "object": "chat.completion",
            "model": response.get("model", ""),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self._map_stop_reason(response.get("stop_reason"))
            }],
            "usage": openai_usage
        }

    def _map_stop_reason(self, anthropic_reason: Optional[str]) -> str:
        """Map Anthropic stop_reason to OpenAI finish_reason."""
        mapping = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls"
        }
        return mapping.get(anthropic_reason, "stop")

    def transform_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform Anthropic SSE chunk to OpenAI SSE format.

        Anthropic events: message_start, content_block_start, content_block_delta,
                         content_block_stop, message_delta, message_stop, error
        OpenAI format: {"choices": [{"delta": {"content": "..."}}]}
        """
        event_type = chunk.get("type")

        if event_type == "content_block_start":
            # Check if this is starting a tool use block
            content_block = chunk.get("content_block", {})
            if content_block.get("type") == "tool_use":
                return {
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": chunk.get("index", 0),
                                "id": content_block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": content_block.get("name", ""),
                                    "arguments": ""
                                }
                            }]
                        },
                        "finish_reason": None
                    }]
                }
            return None

        if event_type == "content_block_delta":
            delta = chunk.get("delta", {})
            if delta.get("type") == "text_delta":
                return {
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta.get("text", "")},
                        "finish_reason": None
                    }]
                }
            elif delta.get("type") == "input_json_delta":
                # Tool call argument streaming
                return {
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": chunk.get("index", 0),
                                "function": {"arguments": delta.get("partial_json", "")}
                            }]
                        },
                        "finish_reason": None
                    }]
                }

        elif event_type == "message_delta":
            # Contains final stop_reason
            stop_reason = chunk.get("delta", {}).get("stop_reason")
            if stop_reason:
                return {
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": self._map_stop_reason(stop_reason)
                    }]
                }

        elif event_type == "message_stop":
            return {
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }

        elif event_type == "error":
            # Surface SSE error events (rate limits, server errors) as finish_reason: "error"
            error_info = chunk.get("error", {})
            return {
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error"
                }],
                "error": {
                    "type": error_info.get("type", "unknown"),
                    "message": error_info.get("message", "Stream error")
                }
            }

        return None  # Skip other event types (message_start, content_block_stop, etc.)

    def parse_sse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse Anthropic SSE line.

        Anthropic uses 'event: <type>' followed by 'data: <json>' format.
        We capture the event type in the returned dict.
        """
        # Handle event type line
        if line.startswith("event: "):
            # Store event type for next data line - caller should handle this
            return {"_event_type": line[7:].strip()}

        if not line.startswith("data: "):
            return None

        data = line[6:]
        if data == "[DONE]":
            return {"done": True}

        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    def get_chat_endpoint(self) -> str:
        return "/messages"  # Anthropic uses /messages not /chat/completions


# Provider registry
PROVIDER_ADAPTERS: Dict[str, ProviderAdapter] = {
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
}


def get_adapter(provider_name: str) -> ProviderAdapter:
    """Get adapter for provider, defaulting to OpenAI-compatible."""
    return PROVIDER_ADAPTERS.get(provider_name, OpenAIAdapter())


def register_adapter(name: str, adapter: ProviderAdapter) -> None:
    """Register a custom adapter for a provider."""
    PROVIDER_ADAPTERS[name] = adapter

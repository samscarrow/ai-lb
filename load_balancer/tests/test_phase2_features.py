"""Tests for Phase 2 features: Oracle consensus, Provider adapters, and Fallback chains."""

import asyncio
import json
import types
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from httpx import RequestError

# Ensure package path for local src
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import app and globals
from load_balancer.main import app as lb_app
from load_balancer import main as lb_main
from load_balancer import config as lb_config
from load_balancer.execution import (
    BackendResult,
    compute_consensus,
)
from load_balancer.providers import (
    OpenAIAdapter,
    AnthropicAdapter,
    get_adapter,
)


def run(coro):
    return asyncio.run(coro)


class FakeRedis:
    def __init__(self):
        self.kv = {}
        self.sets = {"nodes:healthy": set()}

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def scard(self, key):
        return len(self.sets.get(key, set()))

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, val):
        self.kv[key] = val

    async def incrby(self, key, val):
        v = int(self.kv.get(key, 0)) + int(val)
        self.kv[key] = v
        return v

    async def delete(self, key):
        if key in self.kv:
            del self.kv[key]
        return True

    async def expire(self, key, ttl):
        return True

    async def sadd(self, key, val):
        s = self.sets.setdefault(key, set())
        before = len(s)
        s.add(val)
        return 1 if len(s) > before else 0

    async def incrbyfloat(self, key, val):
        v = float(self.kv.get(key, 0.0)) + float(val)
        self.kv[key] = v
        return v

    async def close(self):
        return True


class FakeResponse:
    """Fake response for non-streaming requests."""
    def __init__(self, body, status_code=200, content_type="application/json", headers=None):
        self._body = body
        self.status_code = status_code
        self._content_type = content_type
        self._headers = headers or {}

    @property
    def headers(self):
        h = {"content-type": self._content_type}
        h.update(self._headers)
        return h

    @property
    def content(self):
        if isinstance(self._body, dict):
            return json.dumps(self._body).encode()
        return self._body if isinstance(self._body, bytes) else str(self._body).encode()

    @property
    def text(self):
        return self.content.decode()

    def json(self):
        return json.loads(self.content)


class FakeStreamResponse:
    def __init__(self, chunks, status_code=200, raise_on_enter=None, first_chunk_delay_ms=0, headers=None):
        self._chunks = chunks
        self.status_code = status_code
        self.request = types.SimpleNamespace()
        self._raise_on_enter = raise_on_enter
        self._first_chunk_delay_ms = first_chunk_delay_ms
        self.headers = headers or {}

    async def __aenter__(self):
        if self._raise_on_enter:
            raise self._raise_on_enter
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        first = True
        for c in self._chunks:
            if first and self._first_chunk_delay_ms:
                await asyncio.sleep(self._first_chunk_delay_ms / 1000.0)
            first = False
            yield c


class FakeHTTPClient:
    """Fake HTTP client that supports both .post() and .stream() methods."""
    def __init__(self, behavior):
        # behavior: map node host -> {"response": dict, "error": Exception|None, "status": int, "latency_ms": int}
        self.behavior = behavior
        self.call_order = []

    async def post(self, url, json=None, headers=None, timeout=None):
        # Extract node from URL - handle both local and cloud backends
        if "api.openai.com" in url:
            node = "cloud:openai"
        elif "api.anthropic.com" in url:
            node = "cloud:anthropic"
        else:
            node = url.split("//", 1)[1].split("/", 1)[0]
        self.call_order.append(node)
        cfg = self.behavior.get(node, {"response": {}, "status": 200, "error": None})
        if cfg.get("error"):
            raise cfg["error"]
        latency = cfg.get("latency_ms", 0)
        if latency:
            await asyncio.sleep(latency / 1000.0)
        return FakeResponse(
            cfg.get("response", {}),
            status_code=cfg.get("status", 200),
            headers=cfg.get("headers")
        )

    def stream(self, method, url, json=None, headers=None):
        # Extract node from URL - handle both local and cloud backends
        if "api.openai.com" in url:
            node = "cloud:openai"
        elif "api.anthropic.com" in url:
            node = "cloud:anthropic"
        else:
            node = url.split("//", 1)[1].split("/", 1)[0]
        self.call_order.append(node)
        cfg = self.behavior.get(node, {"chunks": [b""], "status": 200, "error": None})
        if cfg.get("error"):
            return FakeStreamResponse([], raise_on_enter=cfg["error"])
        chunks = cfg.get("chunks")
        if not chunks and cfg.get("response"):
            chunks = [json.dumps(cfg["response"]).encode()]
        return FakeStreamResponse(
            chunks or [b"{}"],
            status_code=cfg.get("status", 200),
            first_chunk_delay_ms=cfg.get("latency_ms", 0),
            headers=cfg.get("headers")
        )

    async def aclose(self):
        return True


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    r = FakeRedis()
    lb_main.redis_client = r
    # Enable multi-exec by default
    lb_config.MULTI_EXEC_ENABLED = True
    lb_config.MULTI_EXEC_MAX_BACKENDS = 3
    lb_config.MULTI_EXEC_TIMEOUT_SECS = 60.0
    lb_config.MULTI_EXEC_CONSENSUS_THRESHOLD = 0.9
    lb_config.BACKEND_ALIASES = {}
    lb_config.BACKEND_ALIASES_REVERSE = {}
    lb_config.CLOUD_BACKENDS = {}
    lb_config.CLOUD_MODELS = {}
    lb_config.FALLBACK_CHAINS = {}
    return r


def make_client():
    return TestClient(lb_app)


# ---------------------- Oracle Consensus Tests ----------------------


class TestOracleConsensus:
    """Test oracle designation in consensus mode."""

    def test_oracle_agrees_with_majority(self):
        """Oracle agrees with majority - no override."""
        response = {"choices": [{"message": {"content": "The answer is 42."}}]}
        results = [
            BackendResult(backend="n1:1234", success=True, response_body=response, latency_ms=100),
            BackendResult(backend="n2:1234", success=True, response_body=response, latency_ms=150),
            BackendResult(backend="cloud:openai", success=True, response_body=response, latency_ms=200),
        ]

        def is_cloud(b):
            return b.startswith("cloud:")

        consensus = compute_consensus(
            results,
            oracle_backend="cloud:openai",
            oracle_present=True,
            is_cloud_fn=is_cloud
        )

        assert consensus.oracle_backend == "cloud:openai"
        assert consensus.oracle_present is True
        assert consensus.oracle_agreed is True
        assert consensus.agreement_count == 3

    def test_oracle_disagrees_overrides_winner(self):
        """Oracle disagrees with majority - oracle becomes winner."""
        majority_response = {"choices": [{"message": {"content": "The answer is 42."}}]}
        oracle_response = {"choices": [{"message": {"content": "The correct answer is 4."}}]}

        results = [
            BackendResult(backend="n1:1234", success=True, response_body=majority_response, latency_ms=100),
            BackendResult(backend="n2:1234", success=True, response_body=majority_response, latency_ms=150),
            BackendResult(backend="cloud:openai", success=True, response_body=oracle_response, latency_ms=200),
        ]

        def is_cloud(b):
            return b.startswith("cloud:")

        consensus = compute_consensus(
            results,
            oracle_backend="cloud:openai",
            oracle_present=True,
            is_cloud_fn=is_cloud
        )

        assert consensus.oracle_backend == "cloud:openai"
        assert consensus.oracle_present is True
        assert consensus.oracle_agreed is False
        # Oracle should override the winner
        assert consensus.winner.backend == "cloud:openai"

    def test_oracle_not_present_no_override(self):
        """Oracle requested but not included - majority wins."""
        response1 = {"choices": [{"message": {"content": "Answer A"}}]}
        response2 = {"choices": [{"message": {"content": "Answer B"}}]}

        results = [
            BackendResult(backend="n1:1234", success=True, response_body=response1, latency_ms=100),
            BackendResult(backend="n2:1234", success=True, response_body=response1, latency_ms=150),
            BackendResult(backend="n3:1234", success=True, response_body=response2, latency_ms=200),
        ]

        consensus = compute_consensus(
            results,
            oracle_backend="cloud:openai",
            oracle_present=False  # Oracle was requested but not available
        )

        assert consensus.oracle_backend == "cloud:openai"
        assert consensus.oracle_present is False
        assert consensus.oracle_agreed is None
        # Winner should be from majority (n1 or n2)
        assert consensus.winner.backend in ("n1:1234", "n2:1234")

    def test_local_cloud_agreement_tracking(self):
        """Track agreement between best local and best cloud."""
        response = {"choices": [{"message": {"content": "Same answer"}}]}

        results = [
            BackendResult(backend="n1:1234", success=True, response_body=response, latency_ms=100),
            BackendResult(backend="n2:1234", success=True, response_body=response, latency_ms=150),
            BackendResult(backend="cloud:openai", success=True, response_body=response, latency_ms=200),
        ]

        def is_cloud(b):
            return b.startswith("cloud:")

        consensus = compute_consensus(
            results,
            is_cloud_fn=is_cloud
        )

        assert consensus.local_cloud_agreement is True
        assert consensus.local_cloud_similarity is not None
        assert consensus.local_cloud_similarity >= 0.9

    def test_local_cloud_disagreement_tracking(self):
        """Track disagreement between best local and best cloud."""
        local_response = {"choices": [{"message": {"content": "Local answer"}}]}
        cloud_response = {"choices": [{"message": {"content": "Cloud answer is different"}}]}

        results = [
            BackendResult(backend="n1:1234", success=True, response_body=local_response, latency_ms=100),
            BackendResult(backend="cloud:openai", success=True, response_body=cloud_response, latency_ms=200),
        ]

        def is_cloud(b):
            return b.startswith("cloud:")

        consensus = compute_consensus(
            results,
            is_cloud_fn=is_cloud
        )

        assert consensus.local_cloud_agreement is False
        assert consensus.local_cloud_similarity is not None
        assert consensus.local_cloud_similarity < 0.9


# ---------------------- Provider Adapter Tests ----------------------


class TestOpenAIAdapter:
    """Test OpenAI adapter."""

    def test_get_headers(self):
        """Test OpenAI header generation."""
        adapter = OpenAIAdapter()
        headers = adapter.get_headers("sk-test-key", {"x-custom": "value"})

        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["x-custom"] == "value"

    def test_passthrough_request(self):
        """Test request passthrough (no transformation needed)."""
        adapter = OpenAIAdapter()
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = adapter.transform_request(body)
        assert result == body

    def test_passthrough_response(self):
        """Test response passthrough (no transformation needed)."""
        adapter = OpenAIAdapter()
        response = {
            "id": "chatcmpl-xxx",
            "choices": [{"message": {"content": "Hello!"}}],
        }
        result = adapter.transform_response(response)
        assert result == response


class TestAnthropicAdapter:
    """Test Anthropic adapter request/response transformation."""

    def test_get_headers(self):
        """Test Anthropic header generation."""
        adapter = AnthropicAdapter()
        headers = adapter.get_headers("sk-ant-test", {"x-custom": "value"})

        assert headers["x-api-key"] == "sk-ant-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "anthropic-beta" in headers
        assert headers["x-custom"] == "value"

    def test_transform_request_basic(self):
        """Test basic request transformation."""
        adapter = AnthropicAdapter()
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "temperature": 0.7,
        }
        result = adapter.transform_request(body)

        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["system"] == "You are helpful."
        assert len(result["messages"]) == 1  # System extracted
        assert result["messages"][0]["role"] == "user"
        assert result["temperature"] == 0.7
        assert "max_tokens" in result

    def test_transform_request_with_tools(self):
        """Test request transformation with tools."""
        adapter = AnthropicAdapter()
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Get weather"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    }
                }
            }]
        }
        result = adapter.transform_request(body)

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert "input_schema" in result["tools"][0]

    def test_transform_response_text(self):
        """Test text response transformation."""
        adapter = AnthropicAdapter()
        anthropic_response = {
            "id": "msg_xxx",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello there!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        result = adapter.transform_response(anthropic_response)

        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello there!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5

    def test_transform_response_tool_use(self):
        """Test tool use response transformation."""
        adapter = AnthropicAdapter()
        anthropic_response = {
            "id": "msg_xxx",
            "model": "claude-sonnet-4-20250514",
            "content": [{
                "type": "tool_use",
                "id": "toolu_xxx",
                "name": "get_weather",
                "input": {"city": "NYC"}
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }
        result = adapter.transform_response(anthropic_response)

        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "NYC"}

    def test_transform_stream_chunk_text_delta(self):
        """Test text streaming chunk transformation."""
        adapter = AnthropicAdapter()
        anthropic_chunk = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"}
        }
        result = adapter.transform_stream_chunk(anthropic_chunk)

        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_transform_stream_chunk_tool_delta(self):
        """Test tool call streaming chunk transformation."""
        adapter = AnthropicAdapter()
        anthropic_chunk = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'}
        }
        result = adapter.transform_stream_chunk(anthropic_chunk)

        assert result is not None
        assert "tool_calls" in result["choices"][0]["delta"]

    def test_transform_stream_chunk_message_stop(self):
        """Test message stop chunk transformation."""
        adapter = AnthropicAdapter()
        anthropic_chunk = {"type": "message_stop"}
        result = adapter.transform_stream_chunk(anthropic_chunk)

        assert result is not None
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_get_chat_endpoint(self):
        """Test Anthropic uses /messages endpoint."""
        adapter = AnthropicAdapter()
        assert adapter.get_chat_endpoint() == "/messages"


class TestAdapterRegistry:
    """Test adapter registry and selection."""

    def test_get_openai_adapter(self):
        """Get OpenAI adapter by name."""
        adapter = get_adapter("openai")
        assert isinstance(adapter, OpenAIAdapter)

    def test_get_anthropic_adapter(self):
        """Get Anthropic adapter by name."""
        adapter = get_adapter("anthropic")
        assert isinstance(adapter, AnthropicAdapter)

    def test_unknown_defaults_to_openai(self):
        """Unknown provider defaults to OpenAI adapter."""
        adapter = get_adapter("unknown_provider")
        assert isinstance(adapter, OpenAIAdapter)


# ---------------------- Fallback Chain Tests ----------------------


class TestFallbackChainConfig:
    """Test fallback chain configuration parsing."""

    def test_parse_simple_chain(self):
        """Parse simple fallback chain."""
        from load_balancer.config import _parse_fallback_chains, FallbackBackend

        chains = _parse_fallback_chains("default=cloud:openai>cloud:anthropic>local:auto")

        assert "default" in chains
        assert len(chains["default"]) == 3
        assert chains["default"][0].backend == "cloud:openai"
        assert chains["default"][1].backend == "cloud:anthropic"
        assert chains["default"][2].backend == "local:auto"

    def test_parse_chain_with_options(self):
        """Parse chain with per-backend options."""
        from load_balancer.config import _parse_fallback_chains

        chains = _parse_fallback_chains("reliable=cloud:openai(timeout=30,retries=2)>cloud:anthropic(timeout=45)")

        assert "reliable" in chains
        assert chains["reliable"][0].backend == "cloud:openai"
        assert chains["reliable"][0].timeout_secs == 30.0
        assert chains["reliable"][0].max_retries == 2
        assert chains["reliable"][1].timeout_secs == 45.0

    def test_parse_multiple_chains(self):
        """Parse multiple chains separated by semicolon."""
        from load_balancer.config import _parse_fallback_chains

        chains = _parse_fallback_chains("fast=cloud:openai>local:auto;reliable=cloud:anthropic>cloud:openai")

        assert "fast" in chains
        assert "reliable" in chains
        assert len(chains["fast"]) == 2
        assert len(chains["reliable"]) == 2


class TestFallbackChainExecution:
    """Test fallback chain HTTP execution."""

    def test_fallback_chain_first_success(self, setup_env):
        """First backend in chain succeeds."""
        r = setup_env

        # Configure cloud backend
        lb_config.CLOUD_BACKENDS = {
            "openai": {
                "url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "provider_type": "openai",
                "is_cloud": True,
            }
        }
        lb_config.CLOUD_MODELS = {"openai": ["gpt-4o"]}
        lb_config.FALLBACK_CHAINS = {
            "default": [
                lb_config.FallbackBackend(backend="cloud:openai", timeout_secs=30, max_retries=1)
            ]
        }

        lb_main.http_client = FakeHTTPClient({
            "cloud:openai": {"response": {"choices": [{"message": {"content": "success"}}]}},
        })

        with make_client() as c:
            resp = c.post(
                "/v1/chat/completions",
                headers={"X-Fallback-Chain": "default"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200
            assert resp.headers.get("x-fallback-chain") == "default"
            assert resp.headers.get("x-fallback-backend") == "cloud:openai"
            assert resp.headers.get("x-fallback-attempts") == "1"

    def test_fallback_chain_failover(self, setup_env):
        """First backend fails, second succeeds."""
        r = setup_env

        # Configure cloud backends
        lb_config.CLOUD_BACKENDS = {
            "openai": {
                "url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "provider_type": "openai",
                "is_cloud": True,
            },
            "anthropic": {
                "url": "https://api.anthropic.com/v1",
                "api_key": "sk-ant-test",
                "provider_type": "anthropic",
                "is_cloud": True,
            }
        }
        lb_config.CLOUD_MODELS = {
            "openai": ["gpt-4o"],
            "anthropic": ["gpt-4o"],  # Pretend it supports the same model
        }
        lb_config.FALLBACK_CHAINS = {
            "reliable": [
                lb_config.FallbackBackend(backend="cloud:openai", timeout_secs=30, max_retries=1),
                lb_config.FallbackBackend(backend="cloud:anthropic", timeout_secs=45, max_retries=1),
            ]
        }

        lb_main.http_client = FakeHTTPClient({
            "cloud:openai": {"status": 500, "response": {"error": "server error"}},
            "cloud:anthropic": {
                "response": {
                    "id": "msg_xxx",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "Fallback success"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5}
                }
            },
        })

        with make_client() as c:
            resp = c.post(
                "/v1/chat/completions",
                headers={"X-Fallback-Chain": "reliable"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200
            assert resp.headers.get("x-fallback-chain") == "reliable"
            assert resp.headers.get("x-fallback-backend") == "cloud:anthropic"
            # Should have tried openai first, then anthropic
            attempts = int(resp.headers.get("x-fallback-attempts", 0))
            assert attempts >= 2

    def test_fallback_chain_takes_precedence_over_execution_mode(self, setup_env):
        """X-Fallback-Chain takes precedence over X-Execution-Mode."""
        r = setup_env

        lb_config.CLOUD_BACKENDS = {
            "openai": {
                "url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "provider_type": "openai",
                "is_cloud": True,
            }
        }
        lb_config.CLOUD_MODELS = {"openai": ["gpt-4o"]}
        lb_config.FALLBACK_CHAINS = {
            "default": [
                lb_config.FallbackBackend(backend="cloud:openai", timeout_secs=30, max_retries=1)
            ]
        }

        lb_main.http_client = FakeHTTPClient({
            "cloud:openai": {"response": {"choices": [{"message": {"content": "ok"}}]}},
        })

        with make_client() as c:
            resp = c.post(
                "/v1/chat/completions",
                headers={
                    "X-Execution-Mode": "consensus",  # Should be ignored
                    "X-Fallback-Chain": "default",    # Takes precedence
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200
            # Should have fallback chain headers, not consensus headers
            assert resp.headers.get("x-fallback-chain") == "default"
            assert "x-execution-mode" not in resp.headers or resp.headers.get("x-execution-mode") != "consensus"

    def test_fallback_chain_429_moves_to_next(self, setup_env):
        """Rate limited (429) backend moves to next in chain."""
        r = setup_env

        lb_config.CLOUD_BACKENDS = {
            "openai": {
                "url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "provider_type": "openai",
                "is_cloud": True,
            },
            "anthropic": {
                "url": "https://api.anthropic.com/v1",
                "api_key": "sk-ant-test",
                "provider_type": "anthropic",
                "is_cloud": True,
            }
        }
        lb_config.CLOUD_MODELS = {"openai": ["gpt-4o"], "anthropic": ["gpt-4o"]}
        lb_config.FALLBACK_CHAINS = {
            "default": [
                lb_config.FallbackBackend(backend="cloud:openai", timeout_secs=30, max_retries=0),
                lb_config.FallbackBackend(backend="cloud:anthropic", timeout_secs=45, max_retries=1),
            ]
        }

        lb_main.http_client = FakeHTTPClient({
            "cloud:openai": {"status": 429, "response": {"error": "rate limited"}, "headers": {"retry-after": "60"}},
            "cloud:anthropic": {
                "response": {
                    "id": "msg_xxx",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "OK"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5}
                }
            },
        })

        with make_client() as c:
            resp = c.post(
                "/v1/chat/completions",
                headers={"X-Fallback-Chain": "default"},
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200
            assert resp.headers.get("x-fallback-backend") == "cloud:anthropic"


# ---------------------- Oracle HTTP Integration Tests ----------------------


class TestOracleHTTPIntegration:
    """Test oracle consensus via HTTP endpoints."""

    def test_consensus_with_valid_oracle_header(self, setup_env):
        """Test consensus mode with valid oracle header."""
        r = setup_env
        r.sets["nodes:healthy"] = {"n1:1234", "n2:1234"}
        run(r.set("node:n1:1234:models", json.dumps({"data": [{"id": "gpt-4o"}]})))
        run(r.set("node:n2:1234:models", json.dumps({"data": [{"id": "gpt-4o"}]})))

        # Configure cloud backend as oracle
        lb_config.CLOUD_BACKENDS = {
            "openai": {
                "url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "provider_type": "openai",
                "is_cloud": True,
            }
        }
        lb_config.CLOUD_MODELS = {"openai": ["gpt-4o"]}

        same_response = {"choices": [{"message": {"content": "42"}}]}
        lb_main.http_client = FakeHTTPClient({
            "n1:1234": {"response": same_response},
            "n2:1234": {"response": same_response},
            "cloud:openai": {"response": same_response},
        })

        with make_client() as c:
            resp = c.post(
                "/v1/chat/completions",
                headers={
                    "X-Execution-Mode": "consensus",
                    "X-Consensus-Oracle": "openai",
                    "X-Max-Backends": "3",
                },
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "2+2?"}],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["mode"] == "consensus"

            # Check oracle headers
            assert resp.headers.get("x-consensus-oracle") == "openai"

    def test_consensus_with_invalid_oracle_header(self, setup_env):
        """Test consensus mode with invalid oracle returns x-consensus-oracle: invalid."""
        r = setup_env
        r.sets["nodes:healthy"] = {"n1:1234", "n2:1234"}
        run(r.set("node:n1:1234:models", json.dumps({"data": [{"id": "m"}]})))
        run(r.set("node:n2:1234:models", json.dumps({"data": [{"id": "m"}]})))

        # No cloud backends configured
        lb_config.CLOUD_BACKENDS = {}

        same_response = {"choices": [{"message": {"content": "42"}}]}
        lb_main.http_client = FakeHTTPClient({
            "n1:1234": {"response": same_response},
            "n2:1234": {"response": same_response},
        })

        with make_client() as c:
            resp = c.post(
                "/v1/chat/completions",
                headers={
                    "X-Execution-Mode": "consensus",
                    "X-Consensus-Oracle": "nonexistent",  # Invalid
                },
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200
            # Should have invalid oracle header
            assert resp.headers.get("x-consensus-oracle") == "invalid"

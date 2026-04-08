"""Tests for multi-backend execution modes (race, all, sequence, consensus)."""

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
    ExecutionMode,
    ExecutionConfig,
    BackendResult,
    ConsensusResult,
    ExecutionEngine,
    compute_consensus,
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
    def __init__(self, body, status_code=200, content_type="application/json"):
        self._body = body
        self.status_code = status_code
        self._content_type = content_type

    @property
    def headers(self):
        return {"content-type": self._content_type}

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
    def __init__(self, chunks, status_code=200, raise_on_enter=None, first_chunk_delay_ms=0):
        self._chunks = chunks
        self.status_code = status_code
        self.request = types.SimpleNamespace()
        self._raise_on_enter = raise_on_enter
        self._first_chunk_delay_ms = first_chunk_delay_ms

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
        node = url.split("//", 1)[1].split("/", 1)[0]
        self.call_order.append(node)
        cfg = self.behavior.get(node, {"response": {}, "status": 200, "error": None})
        if cfg.get("error"):
            raise cfg["error"]
        latency = cfg.get("latency_ms", 0)
        if latency:
            await asyncio.sleep(latency / 1000.0)
        return FakeResponse(cfg.get("response", {}), status_code=cfg.get("status", 200))

    def stream(self, method, url, json=None, headers=None):
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
            first_chunk_delay_ms=cfg.get("latency_ms", 0)
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
    return r


def make_client():
    return TestClient(lb_app)


# ---------------------- Unit Tests for Execution Engine ----------------------


class TestConsensusAlgorithm:
    """Test consensus computation without HTTP."""

    def test_consensus_single_result(self):
        """Single successful result is winner."""
        results = [
            BackendResult(
                backend="n1:1234",
                success=True,
                response_body={"choices": [{"message": {"content": "hello"}}]}
            )
        ]
        consensus = compute_consensus(results)
        assert consensus.winner.backend == "n1:1234"
        assert consensus.agreement_count == 1
        assert not consensus.disagreement

    def test_consensus_unanimous_text(self):
        """All backends agree on text content."""
        response = {"choices": [{"message": {"content": "The answer is 42."}}]}
        results = [
            BackendResult(backend="n1:1234", success=True, response_body=response),
            BackendResult(backend="n2:1234", success=True, response_body=response),
            BackendResult(backend="n3:1234", success=True, response_body=response),
        ]
        consensus = compute_consensus(results)
        assert consensus.agreement_count == 3
        assert not consensus.disagreement

    def test_consensus_majority_text(self):
        """Majority agreement with one dissenting."""
        results = [
            BackendResult(
                backend="n1:1234",
                success=True,
                response_body={"choices": [{"message": {"content": "The answer is 42."}}]}
            ),
            BackendResult(
                backend="n2:1234",
                success=True,
                response_body={"choices": [{"message": {"content": "The answer is 42."}}]}
            ),
            BackendResult(
                backend="n3:1234",
                success=True,
                response_body={"choices": [{"message": {"content": "The answer is 99."}}]}
            ),
        ]
        consensus = compute_consensus(results)
        assert consensus.agreement_count == 2
        assert consensus.disagreement
        assert consensus.winner.backend in ("n1:1234", "n2:1234")

    def test_consensus_tool_calls_unanimous(self):
        """Consensus on identical tool calls."""
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'}
        }
        response = {"choices": [{"message": {"tool_calls": [tool_call]}}]}
        results = [
            BackendResult(backend="n1:1234", success=True, response_body=response),
            BackendResult(backend="n2:1234", success=True, response_body=response),
        ]
        consensus = compute_consensus(results)
        assert consensus.agreement_count == 2
        assert not consensus.disagreement
        assert consensus.comparison_type == "tool_calls"

    def test_consensus_tool_calls_different_args(self):
        """Tool calls with same name but different args = disagreement."""
        tc1 = {"function": {"name": "get_weather", "arguments": '{"city":"NYC"}'}}
        tc2 = {"function": {"name": "get_weather", "arguments": '{"city":"LA"}'}}
        results = [
            BackendResult(
                backend="n1:1234",
                success=True,
                response_body={"choices": [{"message": {"tool_calls": [tc1]}}]}
            ),
            BackendResult(
                backend="n2:1234",
                success=True,
                response_body={"choices": [{"message": {"tool_calls": [tc2]}}]}
            ),
        ]
        consensus = compute_consensus(results)
        assert consensus.agreement_count == 1
        assert consensus.disagreement

    def test_consensus_tool_calls_normalized_json(self):
        """Tool calls with equivalent JSON (different formatting) should agree."""
        tc1 = {"function": {"name": "get_weather", "arguments": '{"city": "NYC", "units": "metric"}'}}
        tc2 = {"function": {"name": "get_weather", "arguments": '{"units":"metric","city":"NYC"}'}}
        results = [
            BackendResult(
                backend="n1:1234",
                success=True,
                response_body={"choices": [{"message": {"tool_calls": [tc1]}}]}
            ),
            BackendResult(
                backend="n2:1234",
                success=True,
                response_body={"choices": [{"message": {"tool_calls": [tc2]}}]}
            ),
        ]
        consensus = compute_consensus(results)
        assert consensus.agreement_count == 2
        assert not consensus.disagreement

    def test_consensus_all_failed(self):
        """All backends failed."""
        results = [
            BackendResult(backend="n1:1234", success=False, error="timeout"),
            BackendResult(backend="n2:1234", success=False, error="connection refused"),
        ]
        consensus = compute_consensus(results)
        assert consensus.agreement_count == 0
        assert consensus.disagreement

    def test_consensus_fastest_wins_in_tie(self):
        """Among agreeing backends, fastest should win."""
        response = {"choices": [{"message": {"content": "42"}}]}
        results = [
            BackendResult(backend="slow:1234", success=True, response_body=response, latency_ms=500),
            BackendResult(backend="fast:1234", success=True, response_body=response, latency_ms=100),
        ]
        consensus = compute_consensus(results)
        assert consensus.winner.backend == "fast:1234"


class TestExecutionEngine:
    """Test ExecutionEngine methods."""

    def test_execute_race_first_wins(self):
        """Race mode returns first successful response."""
        engine = ExecutionEngine()

        async def make_request(backend):
            if backend == "fast:1234":
                return BackendResult(
                    backend=backend,
                    success=True,
                    response_body={"result": "fast"},
                    latency_ms=50
                )
            # Slow backend
            await asyncio.sleep(0.2)
            return BackendResult(
                backend=backend,
                success=True,
                response_body={"result": "slow"},
                latency_ms=200
            )

        async def do_test():
            return await engine.execute_race(
                ["fast:1234", "slow:1234"],
                make_request,
                timeout=5.0
            )

        result = run(do_test())
        assert result.success
        assert result.backend == "fast:1234"

    def test_execute_all_collects_all(self):
        """All mode collects all responses."""
        engine = ExecutionEngine()

        async def make_request(backend):
            return BackendResult(
                backend=backend,
                success=True,
                response_body={"from": backend}
            )

        async def do_test():
            return await engine.execute_all(
                ["n1:1234", "n2:1234", "n3:1234"],
                make_request,
                timeout=5.0
            )

        results = run(do_test())
        assert len(results) == 3
        backends = {r.backend for r in results}
        assert backends == {"n1:1234", "n2:1234", "n3:1234"}

    def test_execute_sequence_in_order(self):
        """Sequence mode executes in order."""
        engine = ExecutionEngine()
        call_order = []

        async def make_request(backend):
            call_order.append(backend)
            return BackendResult(
                backend=backend,
                success=True,
                response_body={"from": backend}
            )

        async def do_test():
            return await engine.execute_sequence(
                ["first:1234", "second:1234", "third:1234"],
                make_request,
                timeout=5.0
            )

        results = run(do_test())
        assert len(results) == 3
        assert call_order == ["first:1234", "second:1234", "third:1234"]

    def test_execute_consensus_returns_result(self):
        """Consensus mode returns ConsensusResult."""
        engine = ExecutionEngine()
        response = {"choices": [{"message": {"content": "42"}}]}

        async def make_request(backend):
            return BackendResult(
                backend=backend,
                success=True,
                response_body=response,
                latency_ms=100
            )

        async def do_test():
            return await engine.execute_consensus(
                ["n1:1234", "n2:1234"],
                make_request,
                timeout=5.0
            )

        result = run(do_test())
        assert isinstance(result, ConsensusResult)
        assert result.agreement_count == 2


# ---------------------- Integration Tests with HTTP Endpoint ----------------------


def test_multi_exec_race_mode(monkeypatch):
    """Test race mode via HTTP header."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234", "n2:1234"}
    run(r.set("node:n1:1234:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:1234:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({
        "n1:1234": {"response": {"choices": [{"message": {"content": "fast"}}]}, "latency_ms": 0},
        "n2:1234": {"response": {"choices": [{"message": {"content": "slow"}}]}, "latency_ms": 100},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "race"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert resp.headers.get("x-execution-mode") == "race"


def test_multi_exec_all_mode(monkeypatch):
    """Test all mode collects responses from all backends."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234", "n2:1234", "n3:1234"}
    for n in ["n1:1234", "n2:1234", "n3:1234"]:
        run(r.set(f"node:{n}:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({
        "n1:1234": {"response": {"choices": [{"message": {"content": "from n1"}}]}},
        "n2:1234": {"response": {"choices": [{"message": {"content": "from n2"}}]}},
        "n3:1234": {"response": {"choices": [{"message": {"content": "from n3"}}]}},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "all"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "all"
        assert data["backends_attempted"] == 3
        assert len(data["responses"]) == 3
        assert resp.headers.get("x-execution-mode") == "all"


def test_multi_exec_consensus_unanimous(monkeypatch):
    """Test consensus mode with unanimous agreement."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234", "n2:1234", "n3:1234"}
    for n in ["n1:1234", "n2:1234", "n3:1234"]:
        run(r.set(f"node:{n}:models", json.dumps({"data": [{"id": "m"}]})))

    # All backends return same content
    same_response = {"choices": [{"message": {"content": "The answer is 42."}}]}
    lb_main.http_client = FakeHTTPClient({
        "n1:1234": {"response": same_response},
        "n2:1234": {"response": same_response},
        "n3:1234": {"response": same_response},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "consensus"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "what is 2+2?"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "consensus"
        assert data["agreement_count"] == 3
        assert data["disagreement"] is False
        assert "x-disagreement" not in resp.headers


def test_multi_exec_consensus_with_disagreement(monkeypatch):
    """Test consensus mode with disagreement."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234", "n2:1234", "n3:1234"}
    for n in ["n1:1234", "n2:1234", "n3:1234"]:
        run(r.set(f"node:{n}:models", json.dumps({"data": [{"id": "m"}]})))

    # Two agree, one disagrees
    lb_main.http_client = FakeHTTPClient({
        "n1:1234": {"response": {"choices": [{"message": {"content": "42"}}]}},
        "n2:1234": {"response": {"choices": [{"message": {"content": "42"}}]}},
        "n3:1234": {"response": {"choices": [{"message": {"content": "99"}}]}},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "consensus"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "what is 2+2?"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "consensus"
        assert data["agreement_count"] == 2
        assert data["disagreement"] is True
        assert resp.headers.get("x-disagreement") == "true"


def test_multi_exec_consensus_tool_calls(monkeypatch):
    """Test consensus mode with tool calls."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234", "n2:1234"}
    for n in ["n1:1234", "n2:1234"]:
        run(r.set(f"node:{n}:models", json.dumps({"data": [{"id": "m"}]})))

    tool_response = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'}
                }]
            }
        }]
    }
    lb_main.http_client = FakeHTTPClient({
        "n1:1234": {"response": tool_response},
        "n2:1234": {"response": tool_response},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "consensus"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "get weather"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agreement_count"] == 2
        assert data["comparison_type"] == "tool_calls"


def test_multi_exec_sequence_mode(monkeypatch):
    """Test sequence mode executes in order."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234", "n2:1234"}
    run(r.set("node:n1:1234:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:1234:models", json.dumps({"data": [{"id": "m"}]})))

    client = FakeHTTPClient({
        "n1:1234": {"response": {"choices": [{"message": {"content": "first"}}]}},
        "n2:1234": {"response": {"choices": [{"message": {"content": "second"}}]}},
    })
    lb_main.http_client = client

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "sequence"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "sequence"
        assert len(data["responses"]) == 2


def test_multi_exec_query_param_fallback(monkeypatch):
    """Test mode can be specified via query param."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234"}
    run(r.set("node:n1:1234:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({
        "n1:1234": {"response": {"choices": [{"message": {"content": "ok"}}]}},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions?mode=race",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-execution-mode") == "race"


def test_multi_exec_disabled_falls_through(monkeypatch):
    """When MULTI_EXEC_ENABLED=False, mode header is ignored."""
    lb_config.MULTI_EXEC_ENABLED = False

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234"}
    run(r.set("node:n1:1234:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({
        "n1:1234": {"chunks": [b'data: {"choices":[{"message":{"content":"single"}}]}\n\n']},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "all"},  # Should be ignored
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        # Should not have multi-exec response format
        body = b"".join(resp.iter_bytes())
        assert b"mode" not in body or b'"mode": "all"' not in body


def test_multi_exec_backend_aliases(monkeypatch):
    """Test backend alias resolution."""
    lb_config.BACKEND_ALIASES = {"m2": "macbook-m2:1234", "m4": "macbook-m4:1234"}
    lb_config.BACKEND_ALIASES_REVERSE = {"macbook-m2:1234": "m2", "macbook-m4:1234": "m4"}

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"macbook-m2:1234", "macbook-m4:1234"}
    run(r.set("node:macbook-m2:1234:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:macbook-m4:1234:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({
        "macbook-m2:1234": {"response": {"choices": [{"message": {"content": "m2"}}]}},
        "macbook-m4:1234": {"response": {"choices": [{"message": {"content": "m4"}}]}},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={
                "X-Execution-Mode": "all",
                "X-Target-Backends": "m2,m4",  # Using aliases
            },
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Check that aliases appear in response
        aliases = [r.get("alias") for r in data["responses"]]
        assert "m2" in aliases or "m4" in aliases


def test_multi_exec_max_backends_limit(monkeypatch):
    """Test X-Max-Backends limits number of backends used."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234", "n2:1234", "n3:1234", "n4:1234"}
    for n in ["n1:1234", "n2:1234", "n3:1234", "n4:1234"]:
        run(r.set(f"node:{n}:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({
        "n1:1234": {"response": {"choices": [{"message": {"content": "1"}}]}},
        "n2:1234": {"response": {"choices": [{"message": {"content": "2"}}]}},
        "n3:1234": {"response": {"choices": [{"message": {"content": "3"}}]}},
        "n4:1234": {"response": {"choices": [{"message": {"content": "4"}}]}},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={
                "X-Execution-Mode": "all",
                "X-Max-Backends": "2",
            },
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["backends_attempted"] == 2
        assert len(data["responses"]) == 2


def test_multi_exec_metrics_recorded(monkeypatch):
    """Test that multi-exec metrics are recorded."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234", "n2:1234"}
    for n in ["n1:1234", "n2:1234"]:
        run(r.set(f"node:{n}:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({
        "n1:1234": {"response": {"choices": [{"message": {"content": "ok"}}]}},
        "n2:1234": {"response": {"choices": [{"message": {"content": "ok"}}]}},
    })

    with make_client() as c:
        c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "consensus"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        # Check metrics were recorded
        total = run(r.get("lb:multi_exec_total"))
        mode_total = run(r.get("lb:multi_exec_total:consensus"))
        assert int(total) >= 1
        assert int(mode_total) >= 1


def test_multi_exec_response_headers(monkeypatch):
    """Test all expected response headers are present."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234", "n2:1234"}
    for n in ["n1:1234", "n2:1234"]:
        run(r.set(f"node:{n}:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({
        "n1:1234": {"response": {"choices": [{"message": {"content": "ok"}}]}, "latency_ms": 100},
        "n2:1234": {"response": {"choices": [{"message": {"content": "ok"}}]}, "latency_ms": 50},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "all"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.headers.get("x-execution-mode") == "all"
        assert "x-backends-attempted" in resp.headers
        assert "x-backends-count" in resp.headers
        assert "x-selected-model" in resp.headers
        assert "x-request-id" in resp.headers
        assert "x-fastest-backend" in resp.headers


def test_multi_exec_race_with_failures(monkeypatch):
    """Test race mode handles backend failures gracefully."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1:1234", "n2:1234"}
    for n in ["n1:1234", "n2:1234"]:
        run(r.set(f"node:{n}:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({
        "n1:1234": {"error": RequestError("connection refused", request=None)},
        "n2:1234": {"response": {"choices": [{"message": {"content": "ok"}}]}},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "race"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        # Should still succeed because n2 worked
        assert resp.status_code == 200

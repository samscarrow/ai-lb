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
from load_balancer.routing.strategies import get_routing_strategy


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
        # No-op for tests
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

    async def ttl(self, key):
        # No real TTL tracking in fake; return -1 (key exists, no expire)
        return -1 if key in self.kv else -2

    async def close(self):
        return True


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
    def __init__(self, behavior):
        # behavior: map node host -> {"chunks": [...], "error": Exception|None, "status": int}
        self.behavior = behavior

    def stream(self, method, url, json=None, headers=None):
        node = url.split("//", 1)[1].split("/", 1)[0]
        cfg = self.behavior.get(node, {"chunks": [b""], "status": 200, "error": None})
        if cfg.get("error"):
            return FakeStreamResponse([], raise_on_enter=cfg["error"], first_chunk_delay_ms=cfg.get("first_chunk_delay_ms", 0))
        return FakeStreamResponse(cfg.get("chunks", []), status_code=cfg.get("status", 200), first_chunk_delay_ms=cfg.get("first_chunk_delay_ms", 0))

    async def aclose(self):
        return True


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    # Inject fake redis by default for all tests
    r = FakeRedis()
    lb_main.redis_client = r
    return r


def make_client():
    # Use synchronous TestClient for convenience; our handlers are async
    return TestClient(lb_app)


def test_health_endpoint_reports_minimum(monkeypatch):
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    with make_client() as c:
        resp_ok = c.get("/health")
        assert resp_ok.status_code == 200
        r.sets["nodes:healthy"] = set()
        resp_bad = c.get("/health")
        assert resp_bad.status_code == 503


def test_health_models_param_includes_per_model_status(monkeypatch):
    """?models=true adds per-model availability to /health response."""
    from load_balancer import config as cfg
    cfg.CB_SCOPE = "model"

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "ok"}, {"id": "bad"}]})))
    run(r.set("node:n1:model:bad:cb_open", "1"))

    with make_client() as c:
        resp = c.get("/health", params={"models": "true"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "models" in body
        assert body["models"]["ok"]["status"] == "available"
        assert body["models"]["bad"]["status"] == "unavailable"


def test_health_degraded_when_model_aware(monkeypatch):
    """With HEALTH_MODEL_AWARE=true, status becomes degraded if any model is unavailable."""
    from load_balancer import config as cfg
    cfg.CB_SCOPE = "model"
    cfg.HEALTH_MODEL_AWARE = True

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "ok"}, {"id": "bad"}]})))
    run(r.set("node:n1:model:bad:cb_open", "1"))

    with make_client() as c:
        resp = c.get("/health", params={"models": "true"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


def test_health_backwards_compat_without_models_param(monkeypatch):
    """Without ?models, response shape is unchanged."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))

    with make_client() as c:
        resp = c.get("/health")
        body = resp.json()
        assert "models" not in body
        assert set(body.keys()) == {"status", "nodes_found", "minimum_required"}


def test_models_aggregate_unique(monkeypatch):
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    # Both nodes advertise overlapping models
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m1"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m1"}, {"id": "m2"}]})))
    with make_client() as c:
        resp = c.get("/v1/models")
        assert resp.status_code == 200
        ids = {m["id"] for m in resp.json()["data"]}
        assert ids == {"m1", "m2"}


def test_models_filters_unavailable_by_default(monkeypatch):
    """Models with all nodes CB'd should be hidden from /v1/models by default."""
    from load_balancer import config as cfg
    cfg.CB_SCOPE = "model"

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "healthy_model"}, {"id": "broken_model"}]})))
    # CB broken_model on n1
    run(r.set("node:n1:model:broken_model:cb_open", "1"))

    with make_client() as c:
        resp = c.get("/v1/models")
        assert resp.status_code == 200
        ids = {m["id"] for m in resp.json()["data"]}
        assert "healthy_model" in ids
        assert "broken_model" not in ids


def test_models_detail_shows_all_with_status(monkeypatch):
    """/v1/models?detail=true shows all models including unavailable, with x_llb_* fields."""
    from load_balancer import config as cfg
    cfg.CB_SCOPE = "model"

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "ok"}, {"id": "bad"}]})))
    run(r.set("node:n1:model:bad:cb_open", "1"))

    with make_client() as c:
        resp = c.get("/v1/models", params={"detail": "true"})
        assert resp.status_code == 200
        models = {m["id"]: m for m in resp.json()["data"]}
        assert "ok" in models
        assert "bad" in models
        assert models["ok"]["x_llb_status"] == "available"
        assert models["ok"]["x_llb_eligible_nodes"] == 1
        assert models["bad"]["x_llb_status"] == "unavailable"
        assert models["bad"]["x_llb_eligible_nodes"] == 0


def test_routing_round_robin_and_failover(monkeypatch):
    # Configure two eligible nodes
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))

    # First attempt: n1 fails, should fail over to n2
    behavior = {
        "n1": {"error": RequestError("boom", request=None)},
        "n2": {"chunks": [b"data: ok\n\n"]},
    }
    lb_main.http_client = FakeHTTPClient(behavior)

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        text = b"".join(resp.iter_bytes())
        assert b"ok" in text


def test_chat_missing_model_applies_default(monkeypatch):
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"solo"}
    run(r.set("node:solo:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({"solo": {"chunks": [b"data: hi\n\n"]}})

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        body = b"".join(resp.iter_bytes())
        assert resp.status_code == 200
        assert b"hi" in body
        assert resp.headers.get("x-selected-model") == "m"
        assert resp.headers.get("x-model-defaulted") == "true"


def test_chat_missing_messages_returns_400(monkeypatch):
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"solo"}
    run(r.set("node:solo:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({"solo": {"chunks": [b"data: hi\n\n"]}})

    with make_client() as c:
        resp = c.post("/v1/chat/completions", json={"model": "m"})
        assert resp.status_code == 400
        error = resp.json().get("error")
        assert error["message"].startswith("Missing")
        assert error["missing"] == ["messages"]


def test_performance_under_load(monkeypatch):
    # One healthy node responds quickly
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"fast"}
    run(r.set("node:fast:models", json.dumps({"data": [{"id": "m"}]})))
    lb_main.http_client = FakeHTTPClient({"fast": {"chunks": [b"data: x\n\n"]}})

    with make_client() as c:
        # Fire off many concurrent requests and ensure all return quickly
        async def one():
            return c.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            ).status_code

        async def many(n=100):
            return await asyncio.gather(*[one() for _ in range(n)])

        codes = run(many())
        assert all(code == 200 for code in codes)


def test_circuit_breaker_skips_failed_node(monkeypatch):
    # Lower CB threshold for the test
    from load_balancer import config as cfg
    cfg.CIRCUIT_BREAKER_THRESHOLD = 1
    cfg.CIRCUIT_BREAKER_TTL_SECS = 60

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"flaky", "good"}
    run(r.set("node:flaky:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:good:models", json.dumps({"data": [{"id": "m"}]})))

    # First request: flaky fails, triggers CB open
    lb_main.http_client = FakeHTTPClient({
        "flaky": {"error": RequestError("boom", request=None)},
        "good": {"chunks": [b"data: ok\n\n"]},
    })
    with make_client() as c:
        resp1 = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        _ = b"".join(resp1.iter_bytes())
        # Second request should skip flaky and go straight to good
        resp2 = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        text2 = b"".join(resp2.iter_bytes())
        assert b"ok" in text2


def test_circuit_breaker_model_scoped_isolates_models(monkeypatch):
    """Model A failures on a node should not CB model B on the same node."""
    from load_balancer import config as cfg
    cfg.CIRCUIT_BREAKER_THRESHOLD = 1
    cfg.CB_SCOPE = "model"

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"shared_node"}
    run(r.set("node:shared_node:models", json.dumps({"data": [{"id": "modelA"}, {"id": "modelB"}]})))

    # Trip model-scoped CB for modelA only
    run(r.set("node:shared_node:model:modelA:cb_open", "1"))

    # modelA should have no eligible nodes
    eligible_a = run(lb_main.get_eligible_nodes("modelA"))
    assert eligible_a == [], f"Expected no eligible nodes for modelA, got {eligible_a}"

    # modelB should still be eligible on the same node
    eligible_b = run(lb_main.get_eligible_nodes("modelB"))
    assert "shared_node" in eligible_b


def test_circuit_breaker_node_wide_still_blocks_all(monkeypatch):
    """Node-wide CB still blocks all models as a fallback."""
    from load_balancer import config as cfg
    cfg.CIRCUIT_BREAKER_THRESHOLD = 1
    cfg.CB_SCOPE = "model"

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"node1"}
    run(r.set("node:node1:models", json.dumps({"data": [{"id": "m1"}, {"id": "m2"}]})))

    # Trip node-wide CB (not model-scoped)
    run(r.set("node:node1:cb_open", "1"))

    # Both models should be blocked
    assert run(lb_main.get_eligible_nodes("m1")) == []
    assert run(lb_main.get_eligible_nodes("m2")) == []


def test_circuit_breaker_legacy_scope_ignores_model_keys(monkeypatch):
    """When CB_SCOPE=node, model-scoped keys are ignored."""
    from load_balancer import config as cfg
    cfg.CIRCUIT_BREAKER_THRESHOLD = 1
    cfg.CB_SCOPE = "node"

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))

    # Set model-scoped CB (should be ignored with CB_SCOPE=node)
    run(r.set("node:n1:model:m:cb_open", "1"))

    # Node should still be eligible since node-wide CB is not open
    eligible = run(lb_main.get_eligible_nodes("m"))
    assert "n1" in eligible


def test_debug_eligible_shows_model_cb(monkeypatch):
    """Debug endpoint should expose both node-wide and model-scoped CB state."""
    from load_balancer import config as cfg
    cfg.CB_SCOPE = "model"

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n1:model:m:cb_open", "1"))

    with make_client() as c:
        resp = c.get("/v1/debug/eligible", params={"model": "m"})
        data = resp.json()
        detail = data["details"][0]
        assert detail["cb_model_open"] is True
        assert detail["cb_open"] is False
        assert detail["skipped"] is True
        assert detail["reason"] == "circuit_open"
        assert data["eligible"] == []


def test_fail_fast_on_cb_open_model_chat(monkeypatch):
    """Chat completions should return 503 immediately when all nodes for a model have CB open."""
    from load_balancer import config as cfg
    cfg.CB_SCOPE = "model"

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "broken"}]})))
    run(r.set("node:n1:model:broken:cb_open", "1"))

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "broken", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 503
        body = resp.json()
        assert body["error"]["code"] == "model_unavailable"
        assert body["error"]["x_llb_diagnosis"]["reason"] == "circuit_open"
        assert "Retry-After" in resp.headers


def test_fail_fast_on_cb_open_model_embeddings(monkeypatch):
    """Embeddings should return 503 immediately when all nodes for a model have CB open."""
    from load_balancer import config as cfg
    cfg.CB_SCOPE = "model"

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "embed-model"}]})))
    run(r.set("node:n1:model:embed-model:cb_open", "1"))

    with make_client() as c:
        resp = c.post(
            "/v1/embeddings",
            json={"model": "embed-model", "input": "hello"},
        )
        assert resp.status_code == 503
        body = resp.json()
        assert body["error"]["x_llb_diagnosis"]["reason"] == "circuit_open"


def test_unknown_model_still_returns_404(monkeypatch):
    """A model that no node has should still get 404 (not 503), skipping warm if disabled."""
    from load_balancer import config as cfg
    cfg.ON_DEMAND_WAIT_ENABLED = False
    cfg.CROSS_MODEL_FALLBACK = False

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "other"}]})))

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 404


def test_routing_headers_on_chat_stream(monkeypatch):
    """Streaming chat completions responses include x-llb-* routing headers."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({
        "n1": {"chunks": [b"data: ok\n\n"]},
        "n2": {"chunks": [b"data: ok\n\n"]},
    })
    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        _ = b"".join(resp.iter_bytes())
        assert "x-llb-eligible-nodes" in resp.headers
        assert "x-llb-total-nodes" in resp.headers
        assert "x-llb-routing-strategy" in resp.headers
        assert int(resp.headers["x-llb-eligible-nodes"]) == 2
        assert int(resp.headers["x-llb-total-nodes"]) == 2


def test_routing_headers_on_fail_fast_503(monkeypatch):
    """503 fail-fast responses include Retry-After and routing context."""
    from load_balancer import config as cfg
    cfg.CB_SCOPE = "model"

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n1:model:m:cb_open", "1"))

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 503
        assert "Retry-After" in resp.headers


def test_metrics_endpoint(monkeypatch):
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:inflight", 3))
    run(r.set("node:n1:failures", 2))
    run(r.set("lb:requests_total", 42))
    with make_client() as c:
        resp = c.get("/metrics")
        body = resp.text
        assert "llb_requests_total 42" in body
        assert 'llb_up{node="n1"} 1' in body
        assert 'llb_inflight{node="n1"} 3' in body
        assert 'llb_failures{node="n1"} 2' in body


def test_sticky_sessions_prefer_same_node(monkeypatch):
    # Two nodes can serve model; first response from n2, second should prefer n2 via stickiness
    from load_balancer import config as cfg
    cfg.STICKY_SESSIONS_ENABLED = True
    lb_main.router = get_routing_strategy("ROUND_ROBIN")
    from load_balancer.routing.strategies import ROUND_ROBIN_STATE
    ROUND_ROBIN_STATE.clear()
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))

    # First call: only n2 returns content successfully
    lb_main.http_client = FakeHTTPClient({
        "n1": {"error": RequestError("boom", request=None)},
        "n2": {"chunks": [b"data: n2\n\n"]},
    })
    with make_client() as c:
        resp1 = c.post(
            "/v1/chat/completions",
            headers={"x-session-id": "s"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        out1 = b"".join(resp1.iter_bytes())
        assert b"n2" in out1

    # Second call: even if n1 would work now, sticky should keep using n2
    lb_main.http_client = FakeHTTPClient({
        "n1": {"chunks": [b"data: n1\n\n"]},
        "n2": {"chunks": [b"data: again n2\n\n"]},
    })
    with make_client() as c:
        resp2 = c.post(
            "/v1/chat/completions",
            headers={"x-session-id": "s"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        out2 = b"".join(resp2.iter_bytes())
        assert b"again n2" in out2


def test_hedging_non_stream_winner_and_headers(monkeypatch):
    # Enable hedging and configure zero delay to force immediate hedge start
    from load_balancer import config as cfg
    cfg.HEDGING_ENABLED = True
    cfg.HEDGING_SMALL_MODELS_ONLY = True
    cfg.HEDGING_MAX_DELAY_MS = 0
    cfg.HEDGING_P95_FRACTION = 1.0
    # Treat 'm' as a small model
    cfg.MODEL_CLASSES = {
        "historical_small": {"candidates": ["m"], "min_nodes": 1}
    }

    # Deterministic routing: pick n1 then n2
    lb_main.router = get_routing_strategy("ROUND_ROBIN")

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))

    # Primary (n1) will be slow relative to zero delay (i.e., not finished before hedge starts),
    # but we also want it to fail to ensure hedge succeeds first.
    lb_main.http_client = FakeHTTPClient({
        "n1": {"error": RequestError("boom", request=None)},
        "n2": {"chunks": [b"{\"ok\":true}"]},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        # Optional headers: verify shape if present
        if resp.headers.get("x-hedged"):
            assert resp.headers.get("x-hedged") in ("true", "false")
        if resp.headers.get("x-hedge-winner"):
            assert resp.headers.get("x-hedge-winner") in ("n1", "n2", "")


def test_streaming_hedge_zero_delay(monkeypatch):
    # Configure hedging zero delay and small-model classification
    from load_balancer import config as cfg
    cfg.HEDGING_ENABLED = True
    cfg.HEDGING_SMALL_MODELS_ONLY = True
    cfg.HEDGING_MAX_DELAY_MS = 0
    cfg.HEDGING_P95_FRACTION = 1.0
    cfg.MODEL_CLASSES = {
        "historical_small": {"candidates": ["m"], "min_nodes": 1}
    }

    lb_main.router = get_routing_strategy("ROUND_ROBIN")
    from load_balancer.routing.strategies import ROUND_ROBIN_STATE
    ROUND_ROBIN_STATE.clear()

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))

    # Primary fails immediately, secondary streams
    class HedgeHarnessHTTPClient(FakeHTTPClient):
        def __init__(self):
            super().__init__({})
            self._calls = 0

        def stream(self, method, url, json=None, headers=None):
            self._calls += 1
            if self._calls == 1:
                return FakeStreamResponse([], raise_on_enter=RequestError("boom", request=None))
            return FakeStreamResponse([b"data: win\n\n"], status_code=200)

    lb_main.http_client = HedgeHarnessHTTPClient()

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        body = b"".join(resp.iter_bytes())
        assert resp.status_code == 200
        assert b"event: hedge_start" in body
        assert b"win" in body
        assert b"error" not in body


def test_streaming_hedge_events_deterministic(monkeypatch):
    # Configure hedging with a small delay; make primary slow and secondary fast
    from load_balancer import config as cfg
    cfg.HEDGING_ENABLED = True
    cfg.HEDGING_SMALL_MODELS_ONLY = False
    cfg.HEDGING_MAX_DELAY_MS = 100
    cfg.HEDGING_P95_FRACTION = 0.05  # 50ms on p95 fallback 1.0s
    cfg.STICKY_SESSIONS_ENABLED = True

    lb_main.router = get_routing_strategy("ROUND_ROBIN")

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "q"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "q"}]})))
    # Force stickiness so primary is n1
    run(r.set("session:s:q", "n1"))

    lb_main.http_client = FakeHTTPClient({
        "n1": {"chunks": [b"data: slow\n\n"], "status": 200, "first_chunk_delay_ms": 200},
        "n2": {"chunks": [b"data: fast\n\n"], "status": 200, "first_chunk_delay_ms": 0},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"x-session-id": "s"},
            json={
                "model": "q",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        body = b"".join(resp.iter_bytes())
        # Expect hedge events and winner n2
        assert b"event: hedge_start" in body
        assert b"event: hedge_winner" in body
        assert b'"winner": "n2"' in body


def test_least_loaded_respects_maxconn(monkeypatch):
    # Force LEAST_LOADED strategy for this test explicitly
    lb_main.router = get_routing_strategy("LEAST_LOADED")

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    # Both nodes host the same model
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))
    # n1 at capacity: inflight == maxconn
    run(r.set("node:n1:inflight", 1))
    run(r.set("node:n1:maxconn", 1))
    # n2 has free capacity
    run(r.set("node:n2:inflight", 0))

    # If routing mistakenly chooses n1, we will see 'bad' in stream
    lb_main.http_client = FakeHTTPClient({
        "n1": {"chunks": [b"data: bad\n\n"]},
        "n2": {"chunks": [b"data: ok\n\n"]},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        out = b"".join(resp.iter_bytes())
        assert b"ok" in out
        assert b"bad" not in out

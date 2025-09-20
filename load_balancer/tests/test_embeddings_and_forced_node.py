import asyncio
import json
import sys
from pathlib import Path
import types
import pytest
from fastapi.testclient import TestClient
from httpx import RequestError

# Ensure package path for local src
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_balancer.main import app as lb_app
from load_balancer import main as lb_main


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


class FakeStreamResponse:
    def __init__(self, chunks, status_code=200, raise_on_enter=None, headers=None):
        self._chunks = chunks
        self.status_code = status_code
        self.request = types.SimpleNamespace()
        self._raise_on_enter = raise_on_enter
        self.headers = headers or {"content-type": "application/json"}

    async def __aenter__(self):
        if self._raise_on_enter:
            raise self._raise_on_enter
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


class FakeHTTPClient:
    def __init__(self, behavior):
        # behavior: map node host -> {"chunks": [...], "error": Exception|None, "status": int, "emb": bytes}
        self.behavior = behavior

    def stream(self, method, url, json=None, headers=None):
        node = url.split("//", 1)[1].split("/", 1)[0]
        cfg = self.behavior.get(node, {"chunks": [b""], "status": 200, "error": None})
        if cfg.get("error"):
            return FakeStreamResponse([], raise_on_enter=cfg["error"])
        return FakeStreamResponse(cfg.get("chunks", []), status_code=cfg.get("status", 200))

    async def post(self, url, json=None, headers=None):
        node = url.split("//", 1)[1].split("/", 1)[0]
        cfg = self.behavior.get(node, {})
        content = cfg.get("emb", b"{}")
        status = cfg.get("status", 200)
        headers = {"content-type": "application/json"}
        return types.SimpleNamespace(content=content, status_code=status, headers=headers, request=types.SimpleNamespace())

    async def aclose(self):
        return True


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    r = FakeRedis()
    lb_main.redis_client = r
    return r


def make_client():
    return TestClient(lb_app)


def test_forced_node_routing_and_errors():
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))

    # Force to n2 returns stream; forcing to unknown raises 404
    lb_main.http_client = FakeHTTPClient({"n2": {"chunks": [b"data: hi\n\n"]}})
    with make_client() as c:
        ok = c.post(
            "/v1/chat/completions?node=n2",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        out = b"".join(ok.iter_bytes())
        assert b"hi" in out
        bad = c.post(
            "/v1/chat/completions?node=unknown",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert bad.status_code == 404


def test_embeddings_success_and_failover():
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"a", "b"}
    run(r.set("node:a:models", json.dumps({"data": [{"id": "em"}]})))
    run(r.set("node:b:models", json.dumps({"data": [{"id": "em"}]})))

    lb_main.http_client = FakeHTTPClient({
        "a": {"status": 500},  # triggers retry
        "b": {"emb": b"{\"data\":[1]}"},
    })
    with make_client() as c:
        resp = c.post(
            "/v1/embeddings",
            headers={"x-request-id": "req-1"},
            json={"model": "em", "input": "hello"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"] == [1]
        # Observability headers present
        assert resp.headers.get("x-selected-model") == "em"
        assert resp.headers.get("x-routed-node") in {"a", "b"}
        assert resp.headers.get("x-request-id") == "req-1"
        attempts = int(resp.headers.get("x-attempts", "1")) if resp.headers.get("x-attempts") else None
        failovers = int(resp.headers.get("x-failover-count", "0")) if resp.headers.get("x-failover-count") else None
        if attempts is not None and failovers is not None:
            assert failovers == max(0, attempts - 1)


def test_embeddings_missing_model_uses_default(monkeypatch):
    from load_balancer import config as cfg

    monkeypatch.setattr(cfg, "DEFAULT_EMBEDDINGS_MODEL", "em")

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"emb"}
    run(r.set("node:emb:models", json.dumps({"data": [{"id": "em"}]})))

    lb_main.http_client = FakeHTTPClient({"emb": {"emb": b"{\"data\":[42]}"}})

    with make_client() as c:
        resp = c.post("/v1/embeddings", json={"input": "hello"})
        assert resp.status_code == 200
        assert resp.json()["data"] == [42]
        assert resp.headers.get("x-selected-model") == "em"
        assert resp.headers.get("x-model-defaulted") == "true"


def test_embeddings_missing_input_returns_400():
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"emb"}
    run(r.set("node:emb:models", json.dumps({"data": [{"id": "em"}]})))

    lb_main.http_client = FakeHTTPClient({"emb": {"emb": b"{\"data\":[1]}"}})

    with make_client() as c:
        resp = c.post("/v1/embeddings", json={"model": "em"})
        assert resp.status_code == 400
        detail = resp.json().get("detail")
        assert detail["missing"] == ["input"]


def test_embeddings_auto_with_intersection_pref(monkeypatch):
    # Two nodes with overlapping models; prefer intersection when require_all=true
    from load_balancer import config as cfg
    cfg.AUTO_MODEL_STRATEGY = "any_first"  # ensure env doesn't force intersection

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"a", "b"}
    # a has em1, em2; b has em2 only
    run(r.set("node:a:models", json.dumps({"data": [{"id": "em1"}, {"id": "em2"}]})))
    run(r.set("node:b:models", json.dumps({"data": [{"id": "em2"}]})))

    # Both nodes can answer em2
    lb_main.http_client = FakeHTTPClient({
        "a": {"emb": b"{\"data\":[2]}"},
        "b": {"emb": b"{\"data\":[2]}"},
    })
    with make_client() as c:
        resp = c.post(
            "/v1/embeddings?require_all=true",
            json={"model": "auto", "input": "hello"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"] == [2]


def test_returns_429_when_saturated():
    # Both nodes at capacity should yield 429 with Retry-After
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"a", "b"}
    run(r.set("node:a:models", json.dumps({"data": [{"id": "em"}]})))
    run(r.set("node:b:models", json.dumps({"data": [{"id": "em"}]})))
    # Set inflight == maxconn for both
    run(r.set("node:a:inflight", 2))
    run(r.set("node:a:maxconn", 2))
    run(r.set("node:b:inflight", 1))
    run(r.set("node:b:maxconn", 1))

    lb_main.http_client = FakeHTTPClient({})
    with make_client() as c:
        resp = c.post(
            "/v1/embeddings",
            headers={"x-request-id": "rid-429"},
            json={"model": "em", "input": "hello"},
        )
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") is not None
        assert resp.headers.get("x-request-id") == "rid-429"

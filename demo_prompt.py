import sys
from pathlib import Path
import asyncio
import types
import json

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "load_balancer" / "src"
sys.path.insert(0, str(SRC))

from load_balancer.main import app as lb_app
from load_balancer import main as lb_main


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

    async def close(self):
        return True


class FakeStreamResponse:
    def __init__(self, chunks, status_code=200, raise_on_enter=None):
        self._chunks = chunks
        self.status_code = status_code
        self.request = types.SimpleNamespace()
        self._raise_on_enter = raise_on_enter

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
        self.behavior = behavior

    def stream(self, method, url, json=None, headers=None):
        node = url.split("//", 1)[1].split("/", 1)[0]
        cfg = self.behavior.get(node, {"chunks": [b"data: (empty)\n\n"]})
        return FakeStreamResponse(cfg.get("chunks", []), status_code=cfg.get("status", 200))

    async def aclose(self):
        return True


def main():
    r = FakeRedis()
    lb_main.redis_client = r
    r.sets["nodes:healthy"] = {"demo:18080"}
    asyncio.get_event_loop().run_until_complete(
        r.set("node:demo:18080:models", json.dumps({"data": [{"id": "demo-model"}]}))
    )

    lb_main.http_client = FakeHTTPClient({
        "demo:18080": {
            "chunks": [
                b"data: Hello, ",
                b"data: world!\n\n",
            ]
        }
    })

    with TestClient(lb_app) as client:
        print("GET /v1/models =>", client.get("/v1/models").json())
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "demo-model", "messages": [{"role": "user", "content": "Hi"}]},
        )
        text = b"".join(resp.iter_bytes()).decode()
        print("\nStreamed response bytes:\n", text)


if __name__ == "__main__":
    main()

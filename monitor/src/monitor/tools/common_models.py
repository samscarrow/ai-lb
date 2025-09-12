import argparse
import asyncio
import json
from typing import List, Set

import httpx
import redis.asyncio as redis

import sys
from pathlib import Path
PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
import config as cfg


async def fetch_models_http(hosts: List[str]) -> List[Set[str]]:
    out: List[Set[str]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        for h in hosts:
            url = f"http://{h}/v1/models"
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            ids = {m.get("id") for m in data.get("data", []) if m.get("id")}
            out.append(ids)
    return out


async def fetch_models_redis(nodes: List[str] | None) -> List[Set[str]]:
    r = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT, decode_responses=True)
    try:
        if not nodes:
            healthy = await r.smembers("nodes:healthy")
            nodes = sorted(list(healthy))
        out: List[Set[str]] = []
        for n in nodes:
            raw = await r.get(f"node:{n}:models")
            if not raw:
                out.append(set())
                continue
            data = json.loads(raw)
            ids = {m.get("id") for m in data.get("data", []) if m.get("id")}
            out.append(ids)
        return out
    finally:
        await r.close()


async def main():
    ap = argparse.ArgumentParser(description="Find exact model ID matches across nodes")
    ap.add_argument("hosts", nargs="*", help="List of host:port entries (e.g., macbook:1234)")
    ap.add_argument("--from-redis", action="store_true", help="Read per-node models from Redis instead of HTTP")
    args = ap.parse_args()

    if args.from_redis:
        sets = await fetch_models_redis(args.hosts or None)
    else:
        if not args.hosts:
            ap.error("hosts are required when not using --from-redis")
        sets = await fetch_models_http(args.hosts)

    if not sets:
        print("No sets fetched.")
        return
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    print(f"Common model IDs across {len(sets)} nodes: {len(common)}")
    for mid in sorted(common):
        print(mid)


if __name__ == "__main__":
    asyncio.run(main())

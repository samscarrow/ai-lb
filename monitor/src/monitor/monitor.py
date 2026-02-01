import asyncio
import httpx
import redis.asyncio as redis
import config

# Set a timeout for network requests
CLIENT_TIMEOUT = httpx.Timeout(30.0)
# Keys should expire if a node is not seen for 2 scan intervals
KEY_EXPIRY_SECONDS = config.SCAN_INTERVAL * 2

async def check_node(redis_client: redis.Redis, session: httpx.AsyncClient, host: str, port: int):
    """
    Checks a single potential node to see if it's a healthy LLM provider.
    If it is, updates its status and model list in Redis.
    """
    node_address = f"{host}:{port}"
    try:
        print(f"Checking node: {node_address}")
        response = await session.get(f"http://{node_address}/v1/models")
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes

        # If we get a successful response, the node is healthy
        models_data = response.json()
        print(f"✅ Found healthy node: {node_address} with {len(models_data.get('data', []))} models.")

        # Create a pipeline to execute Redis commands atomically
        async with redis_client.pipeline() as pipe:
            pipe.sadd("nodes:healthy", node_address)
            pipe.set(f"node:{node_address}:models", response.text)
            # Apply optional per-node concurrency cap
            maxconn = None
            # Explicit map takes precedence
            if node_address in config.MAXCONN_MAP:
                maxconn = config.MAXCONN_MAP[node_address]
            elif config.DEFAULT_MAXCONN:
                maxconn = config.DEFAULT_MAXCONN
            if maxconn and maxconn > 0:
                pipe.set(f"node:{node_address}:maxconn", int(maxconn))
                pipe.expire(f"node:{node_address}:maxconn", KEY_EXPIRY_SECONDS)
            # Set expiry for the keys so they are automatically removed if the monitor fails
            pipe.expire("nodes:healthy", KEY_EXPIRY_SECONDS)
            pipe.expire(f"node:{node_address}:models", KEY_EXPIRY_SECONDS)
            await pipe.execute()

        # Optionally warm configured models to keep them resident/ready
        if config.WARM_ENABLED and config.WARM_MODELS:
            await warm_models(redis_client, session, node_address, config.WARM_MODELS)

    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        # If the node was previously healthy, remove it
        is_healthy = await redis_client.sismember("nodes:healthy", node_address)
        if is_healthy:
            print(f"❌ Node {node_address} is no longer healthy. Removing. Reason: {e}")
            await redis_client.srem("nodes:healthy", node_address)
        else:
            print(f"⚠️ Node {node_address} check failed. Reason: {repr(e)}")
    except Exception as e:
        print(f"An unexpected error occurred while checking {node_address}: {e}")

async def warm_models(redis_client: redis.Redis, session: httpx.AsyncClient, node_address: str, models: list[str]):
    """Attempt a tiny chat completion to ensure the model is loaded and responsive.
    On success, mark node supports:model in Redis so LB can treat it as eligible.
    """
    for model in models:
        ok = False
        for attempt in range(max(1, config.WARM_RETRIES)):
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 4,
                    "stream": False,
                }
                r = await session.post(f"http://{node_address}/v1/chat/completions", json=payload, timeout=httpx.Timeout(config.WARM_TIMEOUT_SECS))
                if r.status_code == 200:
                    ok = True
                    break
            except Exception:
                await asyncio.sleep(0.5)
                continue
        key = f"node:{node_address}:supports:{model}"
        try:
            if ok:
                await redis_client.set(key, 1)
                await redis_client.expire(key, KEY_EXPIRY_SECONDS)
                print(f"🔸 Warmed {model} on {node_address}")
            else:
                # mark unsupported for now (short ttl)
                await redis_client.set(key, 0)
                await redis_client.expire(key, int(KEY_EXPIRY_SECONDS/2))
                print(f"⚠️  Could not warm {model} on {node_address}")
        except Exception:
            pass

async def main():
    """
    Main monitoring loop.
    """
    print("Monitor service started.")
    print(f"Connecting to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}")
    
    redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)

    try:
        # Parse host and port configurations. SCAN_HOSTS supports either plain hosts
        # or host:port pairs. Plain hosts are combined with SCAN_PORTS.
        normalized_hosts = config.SCAN_HOSTS.replace(';', ',').replace(' ', ',')
        raw_hosts = [h.strip() for h in normalized_hosts.split(',') if h.strip()]
        explicit_pairs = []  # list of (host, port)
        hosts_no_port = []
        for h in raw_hosts:
            if ':' in h:
                host, port_str = h.rsplit(':', 1)
                explicit_pairs.append((host, int(port_str)))
            else:
                hosts_no_port.append(h)
        
        normalized_ports = config.SCAN_PORTS.replace(';', ',').replace(' ', ',')
        ports = [int(p.strip()) for p in normalized_ports.split(',') if p.strip()]

        pairs = list(explicit_pairs)
        for h in hosts_no_port:
            for p in ports:
                pairs.append((h, p))

        if not pairs:
            print("Error: No scan targets resolved from SCAN_HOSTS/SCAN_PORTS")
            return
        print(f"Scanning targets: {pairs}")
    except ValueError as e:
        print(f"Error: Invalid host or port configuration: {e}")
        return

    async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as session:
        while True:
            print("--- Starting network scan ---")
            
            # Create a list of tasks for all hosts and ports to check
            tasks = []
            for host, port in pairs:
                tasks.append(check_node(redis_client, session, host, port))
            
            # Run all checks concurrently
            await asyncio.gather(*tasks)
            
            print(f"--- Scan complete. Waiting {config.SCAN_INTERVAL} seconds. ---")
            await asyncio.sleep(config.SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())

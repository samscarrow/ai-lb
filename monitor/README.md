# Monitor Service

This service runs in the background to discover, monitor, and catalog all available LLM nodes on the network. It populates the Redis state store with its findings.

Configuration
- `REDIS_HOST`, `REDIS_PORT`: Redis connection.
- `SCAN_HOSTS`, `SCAN_PORTS`, `SCAN_INTERVAL`: Discovery sweep config.
  - `SCAN_HOSTS` accepts plain hosts (`host.docker.internal,macbook`) OR `host:port` pairs (`host.docker.internal:1234,macbook:1234`).
  - If ports are included in `SCAN_HOSTS`, they take precedence; otherwise each host is paired with all ports from `SCAN_PORTS`.
- `DEFAULT_MAXCONN` (optional): Per-node default concurrency cap applied to all discovered nodes.
- `MAXCONN_MAP` (optional): Comma/semicolon separated `host:port=max` pairs to set per-node caps.
  - Example: `MAXCONN_MAP=host.docker.internal:1234=2,macbook:1234=4`

Behavior
- Healthy nodes are added to `nodes:healthy` and their models stored in `node:{host:port}:models`.
- When `DEFAULT_MAXCONN` or `MAXCONN_MAP` are set, the monitor writes `node:{host:port}:maxconn` so the load balancer’s `LEAST_LOADED` strategy can respect per-node concurrency limits.

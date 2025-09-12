import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# No implicit default; require SCAN_HOSTS to be set (e.g., Tailscale FQDNs or LAN IPs)
SCAN_HOSTS = os.getenv("SCAN_HOSTS", "")
SCAN_PORTS = os.getenv("SCAN_PORTS", "1234")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 60))

# Optional per-node concurrency caps. Two ways to configure:
# 1) `DEFAULT_MAXCONN` applies to all discovered nodes (integer > 0)
# 2) `MAXCONN_MAP` specific caps, format examples:
#    "host:1234=2,macbook:1234=4" or "host:1234=2;macbook:1234=4"
DEFAULT_MAXCONN = int(os.getenv("DEFAULT_MAXCONN", 0)) or None
MAXCONN_MAP_RAW = os.getenv("MAXCONN_MAP", "")

def parse_maxconn_map(raw: str):
    mapping = {}
    for part in [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]:
        if "=" not in part:
            continue
        node, val = part.split("=", 1)
        node = node.strip()
        try:
            mapping[node] = int(val)
        except ValueError:
            continue
    return mapping

MAXCONN_MAP = parse_maxconn_map(MAXCONN_MAP_RAW)

# Optional proactive warming of models to keep them resident/ready
WARM_ENABLED = os.getenv("WARM_ENABLED", "true").lower() in ("1", "true", "yes")
WARM_MODELS_RAW = os.getenv("WARM_MODELS", "")
WARM_MODELS = [m.strip() for m in WARM_MODELS_RAW.replace(";", ",").split(",") if m.strip()]
WARM_TIMEOUT_SECS = int(os.getenv("WARM_TIMEOUT_SECS", 120))
WARM_RETRIES = int(os.getenv("WARM_RETRIES", 1))

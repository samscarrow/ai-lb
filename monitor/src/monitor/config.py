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

# Cloud backend configuration (shared with load balancer)
# Format: "name=url|api_key,name2=url2|api_key2"
# Example: "openai=https://api.openai.com/v1|sk-xxx"
# Cloud backends are registered without probing since cloud APIs don't expose /v1/models the same way
def _parse_cloud_backends(s: str) -> dict:
    """Parse CLOUD_BACKENDS env var into {name: {url: str, api_key: str, is_cloud: True}}."""
    mapping = {}
    for part in [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]:
        if "=" not in part:
            continue
        name, rest = part.split("=", 1)
        name = name.strip()
        rest = rest.strip()
        if "|" not in rest:
            continue
        url, api_key = rest.rsplit("|", 1)
        url = url.strip()
        api_key = api_key.strip()
        if name and url and api_key:
            mapping[name] = {
                "url": url,
                "api_key": api_key,
                "is_cloud": True,
            }
    return mapping

CLOUD_BACKENDS = _parse_cloud_backends(os.getenv("CLOUD_BACKENDS", ""))

# Cloud models configuration - which models to advertise for each cloud backend
# Format: "backend_name=model1,model2,model3;backend_name2=modelA,modelB"
# Example: "openai=gpt-4o,gpt-4o-mini,gpt-3.5-turbo;anthropic=claude-sonnet-4-20250514"
def _parse_cloud_models(s: str) -> dict:
    """Parse CLOUD_MODELS env var into {backend_name: [model1, model2, ...]}."""
    mapping = {}
    for group in [g.strip() for g in s.split(";") if g.strip()]:
        if "=" not in group:
            continue
        name, models_str = group.split("=", 1)
        name = name.strip()
        models = [m.strip() for m in models_str.split(",") if m.strip()]
        if name and models:
            mapping[name] = models
    return mapping

CLOUD_MODELS = _parse_cloud_models(os.getenv("CLOUD_MODELS", ""))

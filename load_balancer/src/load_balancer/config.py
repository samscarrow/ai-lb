import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# Routing strategy
# Supports ROUND_ROBIN, RANDOM, LEAST_LOADED, P2C (Power of Two Choices)
# Default to P2C to improve load spread and tail behavior
ROUTING_STRATEGY = os.getenv("ROUTING_STRATEGY", "P2C")
MIN_HEALTHY_NODES = int(os.getenv("MIN_HEALTHY_NODES", 1))

# Unified LB enhancements
REQUEST_TIMEOUT_SECS = float(os.getenv("REQUEST_TIMEOUT_SECS", 60))
# Harmonize attempts across the codebase: ATTEMPTS_PER_MODEL is the total attempts budget
# (1 initial + retries). MAX_RETRIES is derived for compatibility with existing code paths.
FAILURE_PENALTY_TTL_SECS = int(os.getenv("FAILURE_PENALTY_TTL_SECS", 30))
RETRY_AFTER_SECS = int(os.getenv("RETRY_AFTER_SECS", 2))

# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", 3))
# Deprecated name retained for backward compatibility; prefer CIRCUIT_BREAKER_COOLDOWN_SECS
CIRCUIT_BREAKER_TTL_SECS = int(os.getenv("CIRCUIT_BREAKER_TTL_SECS", 30))

# Model auto-selection configuration
# A model name matching any sentinel will trigger auto-selection.
_RAW = os.getenv("MODEL_SENTINELS", "auto,default")
MODEL_SENTINELS = {s.strip().lower() for s in _RAW.replace(";", ",").split(",") if s.strip()}
LM_MODEL_SENTINEL = os.getenv("LM_MODEL", "auto").strip().lower()
DEFAULT_CHAT_MODEL = (os.getenv("DEFAULT_CHAT_MODEL", "") or "").strip() or None
DEFAULT_EMBEDDINGS_MODEL = (os.getenv("DEFAULT_EMBEDDINGS_MODEL", "") or "").strip() or None
PREFERRED_MODELS = [s.strip() for s in os.getenv("PREFERRED_MODELS", "").replace(";", ",").split(",") if s.strip()]
AUTO_MODEL_STRATEGY = os.getenv("AUTO_MODEL_STRATEGY", "any_first")  # any_first | intersection_first

# Optional static model weights for auto selection (can be overridden via admin API)
def _parse_weights(s: str):
    out = {}
    for part in s.replace(";", ",").split(","):
        if not part.strip():
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            try:
                out[k] = float(v)
            except Exception:
                continue
    return out

MODEL_WEIGHTS = _parse_weights(os.getenv("MODEL_WEIGHTS", ""))

# Enhanced response headers configuration
RESPONSE_HEADERS = [
    "x-selected-model", "x-routed-node", "x-request-id", 
    "x-hedged", "x-hedge-winner", "x-retry-count", "x-fallback-model"
]

# Sticky sessions
STICKY_SESSIONS_ENABLED = os.getenv("STICKY_SESSIONS_ENABLED", "false").lower() in ("1", "true", "yes")
STICKY_TTL_SECS = int(os.getenv("STICKY_TTL_SECS", 600))

# Model classes and fallback configuration
LB_PREFER_SMALL = os.getenv("LM_PREFER_SMALL", "false").lower() in ("1", "true", "yes")
AUTO_MIN_NODES = int(os.getenv("AUTO_MIN_NODES", 2))
STRICT_AUTO_MODE = os.getenv("STRICT_AUTO_MODE", "false").lower() in ("1", "true", "yes")

# Model classes for fallback (can be extended via config)
MODEL_CLASSES = {
    "historical_small": {
        "candidates": ["qwen/qwen3-4b-2507", "liquid/lfm2-1.2b", "mistralai/mistral-small-3.2"],
        "min_nodes": 2
    },
    "historical_medium": {
        "candidates": ["qwen/qwen3-8b", "mlx-community/gpt-oss-20b"],
        "min_nodes": 2
    }
}

# Load model classes from environment if provided
def _parse_model_classes(env_var: str):
    import json
    try:
        return json.loads(os.getenv(env_var, "{}"))
    except Exception:
        return {}

MODEL_CLASSES.update(_parse_model_classes("LB_MODEL_CLASSES"))

# Power of Two Choices routing configuration
P2C_ROUTING_ENABLED = os.getenv("P2C_ROUTING_ENABLED", "true").lower() in ("1", "true", "yes")
P2C_ALPHA = float(os.getenv("P2C_ALPHA", 0.5))  # Weight for p95 latency in scoring
P2C_PENALTY_WEIGHT = float(os.getenv("P2C_PENALTY_WEIGHT", 2.0))  # Weight for recent 5xx rate

# Enhanced retry and failover configuration
ATTEMPTS_PER_MODEL = int(os.getenv("ATTEMPTS_PER_MODEL", 3))
CROSS_MODEL_FALLBACK = os.getenv("CROSS_MODEL_FALLBACK", "false").lower() in ("1", "true", "yes")
RETRY_BACKOFF_MS = [int(x) for x in os.getenv("RETRY_BACKOFF_MS", "50,100").split(",")]

# Derive MAX_RETRIES used in existing flows (attempts after the first)
MAX_RETRIES = max(0, ATTEMPTS_PER_MODEL - 1)

# Circuit breaker enhancements
CIRCUIT_BREAKER_COOLDOWN_SECS = int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECS", 60))
CIRCUIT_BREAKER_SUSPECT_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_SUSPECT_THRESHOLD", 1))
CIRCUIT_BREAKER_SUSPECT_WEIGHT = float(os.getenv("CIRCUIT_BREAKER_SUSPECT_WEIGHT", 0.5))

# Hedging configuration
HEDGING_ENABLED = os.getenv("HEDGING_ENABLED", "true").lower() in ("1", "true", "yes")
HEDGING_SMALL_MODELS_ONLY = os.getenv("HEDGING_SMALL_MODELS_ONLY", "true").lower() in ("1", "true", "yes")
HEDGING_MAX_DELAY_MS = int(os.getenv("HEDGING_MAX_DELAY_MS", 800))
HEDGING_P95_FRACTION = float(os.getenv("HEDGING_P95_FRACTION", 0.6))

# Health and eligibility thresholds
MAX_P95_LATENCY_SECS = float(os.getenv("MAX_P95_LATENCY_SECS", 5.0))
HEALTH_LATENCY_WINDOW_SECS = int(os.getenv("HEALTH_LATENCY_WINDOW_SECS", 120))
ELIGIBILITY_MIN_P95_SAMPLES = int(os.getenv("ELIGIBILITY_MIN_P95_SAMPLES", 20))

# Capacity and fairness
TOKENS_PER_SEC_LIMIT = int(os.getenv("TOKENS_PER_SEC_LIMIT", 0))  # 0 = disabled
FAIRNESS_ENABLED = os.getenv("FAIRNESS_ENABLED", "true").lower() in ("1", "true", "yes")

# On-demand warm/wait when a model has zero eligible nodes
ON_DEMAND_WAIT_ENABLED = os.getenv("ON_DEMAND_WAIT_ENABLED", "true").lower() in ("1", "true", "yes")
ON_DEMAND_WARM_TIMEOUT_SECS = int(os.getenv("ON_DEMAND_WARM_TIMEOUT_SECS", 30))
ON_DEMAND_WARM_GRACE_SECS = int(os.getenv("ON_DEMAND_WARM_GRACE_SECS", 30))
ON_DEMAND_WARM_POLL_MS = int(os.getenv("ON_DEMAND_WARM_POLL_MS", 750))
ON_DEMAND_WARM_FANOUT = int(os.getenv("ON_DEMAND_WARM_FANOUT", 2))  # probe up to N healthy nodes concurrently

# Multi-backend execution configuration
# Enables parallel/sequential execution across multiple API endpoints
MULTI_EXEC_ENABLED = os.getenv("MULTI_EXEC_ENABLED", "true").lower() in ("1", "true", "yes")
MULTI_EXEC_MAX_BACKENDS = int(os.getenv("MULTI_EXEC_MAX_BACKENDS", 3))
MULTI_EXEC_TIMEOUT_SECS = float(os.getenv("MULTI_EXEC_TIMEOUT_SECS", 60.0))
# Consensus similarity threshold for text responses (0.0-1.0)
MULTI_EXEC_CONSENSUS_THRESHOLD = float(os.getenv("MULTI_EXEC_CONSENSUS_THRESHOLD", 0.9))

# Backend aliases for human-friendly names
# Format: "alias1=host1:port1,alias2=host2:port2" or "alias1=host1:port1;alias2=host2:port2"
# Example: "m2=macbook-m2.local:1234,m4=macbook-m4.scarrow.tailnet:1234,gentoo=localhost:11434"
def _parse_backend_aliases(s: str) -> dict:
    """Parse BACKEND_ALIASES env var into {alias: host:port} mapping."""
    mapping = {}
    for part in [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]:
        if "=" not in part:
            continue
        alias, target = part.split("=", 1)
        alias = alias.strip()
        target = target.strip()
        if alias and target:
            mapping[alias] = target
    return mapping

BACKEND_ALIASES = _parse_backend_aliases(os.getenv("BACKEND_ALIASES", ""))

# Reverse mapping: host:port -> alias (for display purposes)
BACKEND_ALIASES_REVERSE = {v: k for k, v in BACKEND_ALIASES.items()}

# Model equivalence groups for multi-backend consensus
# Format: "canonical=model1,model2,model3;another=modelA,modelB"
# Example: "qwen2.5-instruct=qwen2.5-7b-instruct-mlx@4bit,qwen2.5:7b-instruct,qwen2.5-7b-instruct:latest"
# When requesting "qwen2.5-instruct", backends with any of the listed models are eligible
def _parse_model_equivalents(s: str) -> dict:
    """Parse MODEL_EQUIVALENTS into {canonical: [model1, model2, ...]}."""
    mapping = {}
    for group in [g.strip() for g in s.split(";") if g.strip()]:
        if "=" not in group:
            continue
        canonical, models = group.split("=", 1)
        canonical = canonical.strip()
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        if canonical and model_list:
            mapping[canonical] = model_list
    return mapping

MODEL_EQUIVALENTS = _parse_model_equivalents(os.getenv("MODEL_EQUIVALENTS", ""))

# Build reverse lookup: model_name -> canonical name
MODEL_EQUIVALENTS_REVERSE = {}
for canonical, models in MODEL_EQUIVALENTS.items():
    for m in models:
        MODEL_EQUIVALENTS_REVERSE[m] = canonical

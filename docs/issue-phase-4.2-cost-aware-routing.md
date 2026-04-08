# Add cost-aware routing to P2C scoring function

## Overview

Extend the Power of Two Choices (P2C) routing strategy to incorporate estimated per-request cost as a fourth scoring term. When backends have comparable latency and load, the balancer should prefer cheaper options. The cost term is fully additive and opt-in: setting `P2C_BETA=0` (the default) preserves today's behavior exactly. Existing non-P2C strategies (ROUND_ROBIN, RANDOM, LEAST_LOADED) are unaffected.

---

## Component 1: BACKEND_COST_PER_TOKEN Config Schema

### Env Var

**`BACKEND_COST_PER_TOKEN`** — JSON string mapping `model_id` → `{input, output}` prices in USD per million tokens.

```bash
BACKEND_COST_PER_TOKEN='{
  "qwen/qwen3-4b-2507":       {"input": 0.10, "output": 0.30},
  "qwen/qwen3-8b":            {"input": 0.18, "output": 0.54},
  "mlx-community/gpt-oss-20b": {"input": 0.50, "output": 1.50}
}'
```

### JSON Schema

```json
{
  "type": "object",
  "additionalProperties": {
    "type": "object",
    "properties": {
      "input":  { "type": "number", "minimum": 0 },
      "output": { "type": "number", "minimum": 0 }
    },
    "required": ["input", "output"]
  }
}
```

### Python Definition (config.py)

```python
import json
from dataclasses import dataclass

@dataclass
class TokenPricing:
    input: float   # USD per 1M input tokens
    output: float  # USD per 1M output tokens

def _parse_cost_per_token(s: str) -> dict[str, TokenPricing]:
    """Parse BACKEND_COST_PER_TOKEN JSON into {model_id: TokenPricing}."""
    try:
        raw = json.loads(s) if s else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    out = {}
    for model, prices in raw.items():
        if isinstance(prices, dict) and "input" in prices and "output" in prices:
            out[model] = TokenPricing(
                input=float(prices["input"]),
                output=float(prices["output"]),
            )
    return out

BACKEND_COST_PER_TOKEN: dict[str, TokenPricing] = _parse_cost_per_token(
    os.getenv("BACKEND_COST_PER_TOKEN", "")
)
```

### Admin API Override

Add `cost_per_token` to the existing `POST /v1/admin/prefs` handler. Accepts the same JSON structure. On update, replace the module-level `BACKEND_COST_PER_TOKEN` dict.

```python
# In admin prefs handler:
if "cost_per_token" in prefs:
    config.BACKEND_COST_PER_TOKEN = _parse_cost_per_token(
        json.dumps(prefs["cost_per_token"])
    )
```

### Fallback Behavior

When a model has **no pricing entry**, the cost term evaluates to **0.0** (cost-neutral). This means unpriced models are neither penalized nor preferred on cost — they compete purely on load, latency, and failure rate. This is the correct default: it preserves backward compatibility and avoids penalizing models before pricing data is configured.

### Acceptance Criteria

- [ ] `BACKEND_COST_PER_TOKEN='{}'` or unset → `config.BACKEND_COST_PER_TOKEN` is `{}`
- [ ] Valid JSON with 3+ models → parsed into `dict[str, TokenPricing]` correctly
- [ ] Malformed JSON → empty dict, no crash
- [ ] `POST /v1/admin/prefs` with `cost_per_token` field overwrites the config at runtime
- [ ] Models not in the pricing dict produce cost term = 0.0 in scoring

---

## Component 2: Redis Rolling Output-Token Average

### Key Design

```
lb:output_tokens_ewma:{model}|{node}
```

Follows the existing `{model}|{node}` series key convention (see `lb:latency_bucket:{model}|{node}:{le}`).

### Mechanism: EWMA (Exponential Weighted Moving Average)

**Why EWMA over sliding window or sorted set:**
- Single key per series (no unbounded sorted sets, no memory growth)
- O(1) read and write (no ZRANGEBYSCORE scans)
- Self-decaying — stale data loses weight naturally, no TTL-based expiry needed for correctness
- Matches the latency signal's spirit (recent behavior matters more)

### Data Structure

Store two values per series key using a Redis hash:

```
lb:output_tokens_ewma:{model}|{node}  →  { "avg": float, "count": int }
```

With a TTL of `COST_EWMA_TTL_SECS` (default 3600) refreshed on each write, so completely idle series expire.

### Env Vars

| Var | Default | Description |
|-----|---------|-------------|
| `COST_EWMA_ALPHA` | `0.3` | EWMA decay factor. Higher = more weight on recent observations. |
| `COST_EWMA_TTL_SECS` | `3600` | TTL on the EWMA hash key (seconds). |
| `COST_EWMA_COLD_START_TOKENS` | `256` | Default output token estimate when count < `COST_EWMA_MIN_SAMPLES`. |
| `COST_EWMA_MIN_SAMPLES` | `5` | Minimum observations before trusting the EWMA; below this, use cold-start value. |

### Redis Commands

**(a) Recording a new observation** (fire-and-forget after response):

```python
async def _record_output_tokens(model: str, node: str, token_count: int):
    """Update EWMA for output tokens. Fire-and-forget."""
    try:
        key = f"lb:output_tokens_ewma:{model}|{node}"
        alpha = config.COST_EWMA_ALPHA  # 0.3

        vals = await redis_client.hgetall(key)
        old_avg = float(vals.get(b"avg", 0))
        old_count = int(vals.get(b"count", 0))

        if old_count == 0:
            new_avg = float(token_count)
        else:
            new_avg = alpha * token_count + (1 - alpha) * old_avg
        new_count = old_count + 1

        pipe = redis_client.pipeline(transaction=False)
        pipe.hset(key, mapping={"avg": new_avg, "count": new_count})
        pipe.expire(key, config.COST_EWMA_TTL_SECS)
        await pipe.execute()
    except Exception:
        pass
```

**(b) Reading the current average:**

```python
async def _get_output_token_avg(model: str, node: str) -> float:
    """Return EWMA of output tokens, or cold-start default."""
    try:
        key = f"lb:output_tokens_ewma:{model}|{node}"
        vals = await redis_client.hgetall(key)
        count = int(vals.get(b"count", 0))
        if count < config.COST_EWMA_MIN_SAMPLES:
            return float(config.COST_EWMA_COLD_START_TOKENS)
        return float(vals.get(b"avg", config.COST_EWMA_COLD_START_TOKENS))
    except Exception:
        return float(config.COST_EWMA_COLD_START_TOKENS)
```

**(c) Expiry:** Handled by TTL on the hash key (`COST_EWMA_TTL_SECS`). No manual cleanup needed.

### Cold Start

When `count < COST_EWMA_MIN_SAMPLES` (default 5), `_get_output_token_avg` returns `COST_EWMA_COLD_START_TOKENS` (default 256). This is a conservative middle-ground estimate. Since all cold-start model+node pairs return the same value, the cost term cancels out between candidates during cold start — routing degenerates to the existing 3-term formula. This is the desired behavior.

### Acceptance Criteria

- [ ] After 1 observation of 500 tokens, `_get_output_token_avg` returns 256 (cold-start, count < 5)
- [ ] After 5 observations, returns the EWMA value
- [ ] Key has TTL = `COST_EWMA_TTL_SECS` after each write
- [ ] Key expires if no writes for TTL duration
- [ ] Exception in Redis read returns cold-start default (no crash)

---

## Component 3: P2C Scoring Function Update

### New Config

| Var | Default | Description |
|-----|---------|-------------|
| `P2C_BETA` | `0.0` | Weight for cost term. 0 = disabled (backward compatible). Suggested production value: 0.3–0.8. |

```python
# config.py
P2C_BETA = float(os.getenv("P2C_BETA", 0.0))  # Weight for estimated cost in P2C scoring
```

### Normalization Strategy

The cost term must be in [0, 1] to be comparable with the other terms. We use **min-max normalization across the two P2C candidates** at scoring time:

```
estimated_cost = output_token_avg * (pricing.output / 1_000_000)
```

Since P2C always compares exactly 2 candidates, we compute raw cost for both, then normalize:

```
cost_min = min(cost_a, cost_b)
cost_max = max(cost_a, cost_b)
range = cost_max - cost_min

if range == 0:
    cost_normalized = 0.0  # both equal → no preference
else:
    cost_normalized = (cost_raw - cost_min) / range  # cheaper=0, pricier=1
```

This avoids the need for a global max-cost config and adapts automatically to the candidate pair.

### Updated Formula

```
score = inflight_normalized + (alpha * p95_latency) + (penalty_weight * failure_rate) + (beta * cost_normalized)
```

### Updated Pseudocode

The scoring must change from a single-node method to a two-phase approach: compute raw costs for all candidates first, then normalize and score.

```python
class PowerOfTwoChoicesStrategy(RoutingStrategy):

    def __init__(self, alpha: float = 0.5, penalty_weight: float = 2.0, beta: float = 0.0):
        self.alpha = alpha
        self.penalty_weight = penalty_weight
        self.beta = beta

    async def select_node(self, nodes, model_name, redis_client):
        if not nodes:
            return None
        if len(nodes) == 1:
            return nodes[0]

        candidates = random.sample(nodes, min(2, len(nodes)))

        # Phase 1: collect base scores and raw costs
        base_scores = {}
        raw_costs = {}
        for node in candidates:
            base_scores[node] = await self._calculate_base_score(node, model_name, redis_client)
            if self.beta > 0:
                raw_costs[node] = await self._estimate_raw_cost(node, model_name, redis_client)
            else:
                raw_costs[node] = 0.0

        # Phase 2: normalize costs across candidates
        cost_vals = list(raw_costs.values())
        c_min, c_max = min(cost_vals), max(cost_vals)
        c_range = c_max - c_min

        best_node, best_score = None, float('inf')
        for node in candidates:
            cost_norm = ((raw_costs[node] - c_min) / c_range) if c_range > 0 else 0.0
            score = base_scores[node] + (self.beta * cost_norm)
            if score < best_score:
                best_score = score
                best_node = node
        return best_node

    async def _calculate_base_score(self, node, model_name, redis_client) -> float:
        """Original 3-term score (inflight + latency + failures)."""
        try:
            inflight_val = await redis_client.get(f"node:{node}:inflight")
            maxconn_val = await redis_client.get(f"node:{node}:maxconn")
            inflight = int(inflight_val) if inflight_val is not None else 0
            maxconn = int(maxconn_val) if maxconn_val not in (None, "", "0") else 100
            inflight_normalized = inflight / maxconn

            series_key = f"{model_name}|{node}"
            p95_latency = await self._get_p95_latency(series_key, redis_client)
            failure_rate = await self._get_failure_rate(node, redis_client)

            return inflight_normalized + (self.alpha * p95_latency) + (self.penalty_weight * failure_rate)
        except Exception:
            return float('inf')

    async def _estimate_raw_cost(self, node, model_name, redis_client) -> float:
        """Estimated cost in USD for a request to this model+node."""
        from .. import config as cfg
        pricing = cfg.BACKEND_COST_PER_TOKEN.get(model_name)
        if not pricing:
            return 0.0  # No pricing → cost-neutral
        avg_output = await _get_output_token_avg(model_name, node)
        # Output-token cost dominates; input tokens are same across candidates for same request
        return avg_output * (pricing.output / 1_000_000)
```

### Edge Cases

- **`P2C_BETA=0.0`** (default): cost term is always 0. No Redis reads for EWMA. No behavior change.
- **No pricing for model**: `_estimate_raw_cost` returns 0.0. Both candidates get 0.0 raw cost → `c_range=0` → `cost_norm=0.0`. Neutral.
- **Cold start (< 5 observations)**: Both candidates use the same cold-start default → `c_range=0` → neutral.
- **Single node**: Short-circuit returns the node. No scoring at all.

### Constructor Wiring (strategies.py `get_routing_strategy`)

```python
if strategy_name.upper() in ("P2C", "POWER_OF_TWO"):
    from .. import config
    return strategy_class(
        alpha=kwargs.get("alpha", config.P2C_ALPHA),
        penalty_weight=kwargs.get("penalty_weight", config.P2C_PENALTY_WEIGHT),
        beta=kwargs.get("beta", config.P2C_BETA),
    )
```

### Acceptance Criteria

- [ ] `P2C_BETA=0` → scoring function produces identical results to current implementation
- [ ] `P2C_BETA=0.5` with pricing for model → cheaper node gets lower score
- [ ] Two candidates with identical base scores but different costs → cheaper node wins
- [ ] Two candidates with no pricing data → tie on cost term (both 0.0)
- [ ] `ComplexityRoutingStrategy` (which delegates to P2C internally) inherits cost awareness automatically

---

## Component 4: Write-Path Token Accounting

### Instrumentation Points in main.py

**Non-streaming path** — `attempt_request()` (line ~2576), after the response is received and validated:

```python
# After line 2649 (out = Response(...)), before return:
if config.P2C_BETA > 0:
    try:
        resp_json = json.loads(content)
        usage = resp_json.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is not None:
            asyncio.create_task(_record_output_tokens(model_name, node, int(completion_tokens)))
        else:
            logger.debug("[req=%s] No usage.completion_tokens in response", request_id)
    except (json.JSONDecodeError, Exception):
        pass
```

**Streaming path** — `attempt_stream()` (line ~2130). Two sub-cases:

**(a) Passthrough (local/OpenAI-compatible backends)** — line ~2214. The final SSE chunk with `stream_options.include_usage=true` contains a `usage` object. Accumulate chunks and check:

```python
# Inside the passthrough loop (line 2216):
_stream_token_count = 0
_stream_usage_found = False
async for chunk in response.aiter_bytes():
    yield chunk
    # Best-effort: parse SSE data lines for usage
    if config.P2C_BETA > 0 and not _stream_usage_found:
        try:
            text = chunk.decode("utf-8", errors="replace")
            for line in text.split("\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    payload = json.loads(line[6:])
                    usage = payload.get("usage")
                    if usage and "completion_tokens" in usage:
                        _stream_token_count = usage["completion_tokens"]
                        _stream_usage_found = True
        except Exception:
            pass

# After the stream loop (in the finally/post-yield section around line 2218):
if config.P2C_BETA > 0 and _stream_token_count > 0:
    asyncio.create_task(_record_output_tokens(model_name, node, _stream_token_count))
```

**(b) Cloud adapter path** — line ~2182. Same approach: check each transformed chunk for a `usage` field before yielding.

### Fire-and-Forget

All `_record_output_tokens` calls use `asyncio.create_task()` — they run in the background and do not block the response. Exceptions inside are caught silently (per the function's try/except).

### Missing Usage Data

- Non-streaming: if `usage.completion_tokens` is absent, log at DEBUG level, skip recording. No error.
- Streaming: if no chunk contains `usage`, `_stream_token_count` stays 0, no recording. No error.
- This means the EWMA will simply have fewer observations and stay on the cold-start default longer. Acceptable: local backends that don't report usage will be cost-neutral.

### Request Body: Enable Usage Reporting

For streaming requests to backends that support it, inject `stream_options.include_usage = true` if not already present:

```python
# Before sending the streaming request:
if config.P2C_BETA > 0 and is_stream:
    req_body = {**req_body}  # shallow copy
    req_body.setdefault("stream_options", {})
    req_body["stream_options"].setdefault("include_usage", True)
```

### Acceptance Criteria

- [ ] Non-streaming response with `usage.completion_tokens=150` → EWMA key updated with 150
- [ ] Non-streaming response without `usage` field → no EWMA write, no error
- [ ] Streaming response final chunk with `usage.completion_tokens=300` → EWMA key updated
- [ ] Streaming response with no usage chunks → no EWMA write, no error
- [ ] `P2C_BETA=0` → no usage parsing, no EWMA writes (zero overhead)
- [ ] Token recording does not add measurable latency to response delivery

---

## Testing Strategy

1. **Config parsing unit test**: Set `BACKEND_COST_PER_TOKEN` to valid JSON with 3 models. Assert `config.BACKEND_COST_PER_TOKEN` has 3 entries with correct `TokenPricing` values. Set to malformed JSON. Assert empty dict.

2. **EWMA unit test**: Mock Redis. Call `_record_output_tokens("modelA", "node1", 500)` 6 times. Assert `_get_output_token_avg("modelA", "node1")` returns EWMA (not cold-start). Assert count=6. Verify TTL was set.

3. **Cold-start test**: Mock Redis with empty keys. Assert `_get_output_token_avg` returns `COST_EWMA_COLD_START_TOKENS` (256).

4. **P2C scoring test (cost disabled)**: Set `P2C_BETA=0`. Mock two nodes with different inflight/latency. Assert selected node matches the existing (non-cost) scoring logic exactly.

5. **P2C scoring test (cost enabled)**: Set `P2C_BETA=0.5`. Mock two nodes with identical inflight/latency/failures. Set pricing so node A costs 2× node B. Assert node B is selected.

6. **P2C scoring test (no pricing)**: Set `P2C_BETA=0.5` but no entry in `BACKEND_COST_PER_TOKEN` for the model. Assert both nodes score identically on cost (0.0).

7. **Non-streaming token extraction**: Send a mock response with `usage.completion_tokens=200`. Assert `_record_output_tokens` was called with 200.

8. **Streaming token extraction**: Send mock SSE chunks where the last chunk contains `usage.completion_tokens=350`. Assert `_record_output_tokens` was called with 350.

9. **Admin API override**: POST to `/v1/admin/prefs` with `cost_per_token` payload. Assert `config.BACKEND_COST_PER_TOKEN` updated. Subsequent P2C scoring uses new values.

10. **Backward compatibility**: Deploy with zero cost-related env vars. Run the full existing test suite. Assert all tests pass with no behavior change.

---

## Configuration Summary

| Env Var | Default | Description |
|---------|---------|-------------|
| `BACKEND_COST_PER_TOKEN` | `""` (empty) | JSON mapping model → `{input, output}` prices in USD/1M tokens |
| `P2C_BETA` | `0.0` | Weight for cost term in P2C scoring. 0 = disabled. |
| `COST_EWMA_ALPHA` | `0.3` | EWMA decay factor for output-token rolling average |
| `COST_EWMA_TTL_SECS` | `3600` | TTL for EWMA Redis keys (seconds) |
| `COST_EWMA_COLD_START_TOKENS` | `256` | Default output token estimate during cold start |
| `COST_EWMA_MIN_SAMPLES` | `5` | Min observations before trusting EWMA over cold-start default |

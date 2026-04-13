import abc
import time
import json
import random
import math
from typing import List, Optional, Dict, Any
from itertools import cycle

from .. import config as _config

# In-memory state for round-robin cycling
# A dictionary to hold an iterator for each model
ROUND_ROBIN_STATE = {}

class RoutingStrategy(abc.ABC):
    """Abstract base class for all routing strategies."""

    @abc.abstractmethod
    async def select_node(self, nodes: List[str], model_name: str, redis_client) -> str:
        """Selects a node from the available list. Returns a node address (host:port)."""
        pass

class RoundRobinStrategy(RoutingStrategy):
    """Cycles through available nodes for a given model in a round-robin fashion."""

    async def select_node(self, nodes: List[str], model_name: str, redis_client) -> str:
        if not nodes:
            return None

        # Get or create the cycle iterator for this model
        if model_name not in ROUND_ROBIN_STATE or set(ROUND_ROBIN_STATE[model_name]['nodes']) != set(nodes):
            ROUND_ROBIN_STATE[model_name] = {
                'nodes': nodes,
                'iterator': cycle(nodes)
            }
        
        # Get the next node from the cycle
        return next(ROUND_ROBIN_STATE[model_name]['iterator'])

class RandomStrategy(RoutingStrategy):
    """Selects a node randomly from the available list."""

    async def select_node(self, nodes: List[str], model_name: str, redis_client) -> str:
        if not nodes:
            return None
        return random.choice(nodes)

class LeastLoadedStrategy(RoutingStrategy):
    """Selects the node with the fewest in-flight requests (from Redis counters)."""

    async def select_node(self, nodes: List[str], model_name: str, redis_client) -> str:
        if not nodes:
            return None
        best_node = None
        best_load = None
        # Use integer counters: key=node:{node}:inflight
        # Respect optional per-node concurrency caps via key node:{node}:maxconn
        eligible = []
        for n in nodes:
            try:
                inflight_val = await redis_client.get(f"node:{n}:inflight")
                maxconn_val = await redis_client.get(f"node:{n}:maxconn")
                load = int(inflight_val) if inflight_val is not None else 0
                maxconn = int(maxconn_val) if maxconn_val not in (None, "", "0") else None
            except Exception:
                load = 0
                maxconn = None
            # Skip nodes that are at/over their max concurrency
            if maxconn is not None and load >= maxconn:
                continue
            eligible.append((n, load))

        # If all nodes are saturated, return None to signal no capacity
        if not eligible:
            return None

        for n, load in eligible:
            if best_load is None or load < best_load:
                best_load = load
                best_node = n
        return best_node


class PowerOfTwoChoicesStrategy(RoutingStrategy):
    """Power of Two Choices routing: sample 2 eligible nodes, pick the one with lower score.

    Supports two scoring modes:
    - **Peak EWMA** (default, P2C_PEAK_EWMA=true): Finagle/Envoy-inspired multiplicative
      scoring: score = ewma_rtt * (pending + 1). Dimensionally consistent, no tunable weights.
      Failures inject penalty RTT that decays naturally.
    - **Legacy additive** (P2C_PEAK_EWMA=false): score = inflight/maxconn + alpha*p95 + penalty*failures.

    Cost term (beta > 0) is additive on top of either mode.
    """

    def __init__(self, alpha: float = 0.5, penalty_weight: float = 2.0, beta: float = 0.0):
        self.alpha = alpha
        self.penalty_weight = penalty_weight
        self.beta = beta
        self.use_peak_ewma = getattr(_config, "P2C_PEAK_EWMA", True)

    async def select_node(self, nodes: List[str], model_name: str, redis_client) -> Optional[str]:
        if not nodes:
            return None
        if len(nodes) == 1:
            return nodes[0]

        # Sample 2 nodes randomly
        candidates = random.sample(nodes, min(2, len(nodes)))

        # Calculate scores for each candidate
        if self.use_peak_ewma:
            scores = {}
            for node in candidates:
                scores[node] = await self._peak_ewma_score(node, model_name, redis_client)
        else:
            scores = {}
            for node in candidates:
                scores[node] = await self._legacy_score(node, model_name, redis_client)

        # Add cost term with min-max normalization across candidates (opt-in via beta > 0)
        if self.beta > 0:
            costs = {node: await self._get_cost_estimate(node, model_name, redis_client) for node in candidates}
            cost_vals = list(costs.values())
            cost_min, cost_max = min(cost_vals), max(cost_vals)
            cost_range = cost_max - cost_min
            for node in candidates:
                cost_norm = (costs[node] - cost_min) / cost_range if cost_range > 0 else 0.0
                scores[node] += self.beta * cost_norm

        return min(scores, key=lambda n: scores[n])

    # ---------- Peak EWMA scoring (Finagle/Envoy, Apache-2.0) ----------

    async def _peak_ewma_score(self, node: str, model_name: str, redis_client) -> float:
        """Score = ewma_rtt * (pending + 1).

        Inspired by twitter/finagle PeakEwma.scala and envoyproxy/envoy PR #40653.
        Multiplicative composition means units are request-seconds (Little's Law).
        No tunable weights — latency and load naturally balance.
        """
        try:
            # Get current EWMA RTT (with time decay applied)
            ewma_rtt = await self._get_ewma_rtt(node, model_name, redis_client)

            # Get pending (inflight) requests
            inflight_val = await redis_client.get(f"node:{node}:inflight")
            pending = int(inflight_val) if inflight_val is not None else 0

            return ewma_rtt * (pending + 1)
        except Exception:
            return float("inf")

    async def _get_ewma_rtt(self, node: str, model_name: str, redis_client) -> float:
        """Retrieve EWMA RTT with time-based decay applied at read time.

        If no data exists, returns PEAK_EWMA_DEFAULT_RTT (cold start).
        Stale data decays toward zero, naturally encouraging probe traffic
        to nodes that haven't been sampled recently.
        """
        key_rtt = f"lb:ewma_rtt:{model_name}|{node}"
        key_ts = f"lb:ewma_ts:{model_name}|{node}"
        default_rtt = getattr(_config, "PEAK_EWMA_DEFAULT_RTT", 0.5)
        decay = getattr(_config, "PEAK_EWMA_DECAY_SECS", 10.0)

        rtt_raw = await redis_client.get(key_rtt)
        ts_raw = await redis_client.get(key_ts)

        if rtt_raw is None or ts_raw is None:
            return default_rtt

        stored_rtt = float(rtt_raw)
        stored_ts = float(ts_raw)
        now = time.time()
        dt = max(now - stored_ts, 0.0)

        # Apply decay: the older the measurement, the more it shrinks toward zero
        # This encourages the balancer to probe nodes it hasn't heard from recently
        decay_factor = math.exp(-dt / decay) if decay > 0 else 0.0
        return stored_rtt * decay_factor

    # ---------- Legacy additive scoring ----------

    async def _legacy_score(self, node: str, model_name: str, redis_client) -> float:
        """Legacy additive score: inflight/maxconn + alpha*p95 + penalty*failures."""
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
            return float("inf")

    async def _get_p95_latency(self, series_key: str, redis_client) -> float:
        """Calculate approximate p95 latency from histogram buckets."""
        try:
            buckets = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
            total_count = 0
            cumulative_counts = []

            for le in buckets:
                val = await redis_client.get(f"lb:latency_bucket:{series_key}:{le}")
                count = int(val) if val else 0
                cumulative_counts.append(count)
                if le == float("inf"):
                    total_count = count

            if total_count == 0:
                return 0.0

            p95_target = total_count * 0.95

            for i, count in enumerate(cumulative_counts):
                if count >= p95_target:
                    if i == 0:
                        return buckets[i] * 0.5
                    lower_bound = buckets[i-1] if i > 0 else 0
                    upper_bound = buckets[i]
                    if i < len(cumulative_counts) - 1:
                        prev_count = cumulative_counts[i-1] if i > 0 else 0
                        bucket_range = count - prev_count
                        if bucket_range > 0:
                            position = (p95_target - prev_count) / bucket_range
                            return lower_bound + position * (upper_bound - lower_bound)
                    return upper_bound if upper_bound != float("inf") else 10.0

            return 0.0
        except Exception:
            return 0.0

    # ---------- Shared helpers ----------

    async def _get_cost_estimate(self, node: str, model_name: str, redis_client) -> float:
        """Return estimated cost (USD) for one request to node based on EWMA output tokens."""
        try:
            pricing = getattr(_config, "BACKEND_COST_PER_TOKEN", {}).get(model_name)
            if not pricing:
                return 0.0
            output_price = float(pricing.get("output", 0.0))
            if output_price <= 0:
                return 0.0
            cold_start = float(getattr(_config, "COST_EWMA_COLD_START_TOKENS", 256))
            min_samples = int(getattr(_config, "COST_EWMA_MIN_SAMPLES", 5))
            key_ewma = f"lb:output_tokens_ewma:{model_name}|{node}"
            key_count = f"lb:output_tokens_count:{model_name}|{node}"
            ewma_val = await redis_client.get(key_ewma)
            count_val = await redis_client.get(key_count)
            count = int(count_val) if count_val else 0
            tokens = float(ewma_val) if (ewma_val and count >= min_samples) else cold_start
            return tokens * output_price / 1_000_000.0
        except Exception:
            return 0.0

    async def _get_failure_rate(self, node: str, redis_client) -> float:
        """Get recent failure rate for the node (legacy scoring only)."""
        try:
            failures = await redis_client.get(f"node:{node}:failures")
            failure_count = int(failures) if failures else 0
            return min(failure_count / 10.0, 1.0)
        except Exception:
            return 0.0


class ConsistentHashingStrategy(RoutingStrategy):
    """Consistent hashing for sticky sessions."""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self._hash_ring: Dict[int, str] = {}
        self._nodes: List[str] = []
        
    async def select_node(self, nodes: List[str], model_name: str, redis_client, session_id: Optional[str] = None) -> Optional[str]:
        if not nodes:
            return None
            
        # If no session_id, fall back to random selection
        if not session_id:
            return random.choice(nodes)
            
        # Rebuild hash ring if nodes changed
        if set(nodes) != set(self._nodes):
            self._rebuild_hash_ring(nodes)
            
        # Find node for session
        return self._get_node_for_session(session_id)
        
    def _rebuild_hash_ring(self, nodes: List[str]):
        """Rebuild the consistent hash ring."""
        self._hash_ring.clear()
        self._nodes = nodes[:]
        
        for node in nodes:
            for i in range(self.virtual_nodes):
                virtual_key = f"{node}:{i}"
                hash_value = hash(virtual_key) % (2**32)
                self._hash_ring[hash_value] = node
                
    def _get_node_for_session(self, session_id: str) -> Optional[str]:
        """Get the node responsible for a session."""
        if not self._hash_ring:
            return None
            
        session_hash = hash(session_id) % (2**32)
        
        # Find the first node clockwise from the session hash
        sorted_hashes = sorted(self._hash_ring.keys())
        for hash_value in sorted_hashes:
            if hash_value >= session_hash:
                return self._hash_ring[hash_value]
                
        # Wrap around to the first node
        return self._hash_ring[sorted_hashes[0]]


class ComplexityRoutingStrategy(RoutingStrategy):
    """Complexity-aware routing strategy.

    For node selection it delegates to P2C (best performance within the tier).
    The real value is in the scoring API used by main.py for model tier selection:
    score_prompt_complexity() + get_complexity_model() let the router classify
    a prompt into small / medium / large tiers before calling get_eligible_nodes().

    Inspired by RouteLLM (Apache-2.0) — lm-sys/RouteLLM
    Adaptation: heuristic scorer (no extra model call), wired into MODEL_CLASSES tiers.
    """

    # Complexity thresholds: [0, LOW) → small, [LOW, HIGH) → medium, [HIGH, 1] → large/any
    # Calibrated from AILB-MT-1 telemetry (2026-03-04): scorer ceiling is ~0.32,
    # so the original 0.35/0.65 bounds were unreachable. New values use the full
    # observed score range: small<0.10, medium 0.10–0.25, large>0.25.
    LOW_THRESHOLD: float = 0.10
    HIGH_THRESHOLD: float = 0.25

    def __init__(self):
        self._p2c = PowerOfTwoChoicesStrategy()

    async def select_node(self, nodes: List[str], model_name: str, redis_client) -> Optional[str]:
        return await self._p2c.select_node(nodes, model_name, redis_client)

    @staticmethod
    def score_prompt_complexity(messages: List[Dict]) -> float:
        """Score prompt complexity from OpenAI-format messages. Returns [0.0, 1.0].

        Signals:
          - Character count:    weight 0.30  linear scale, cap at 10 000 chars
          - Code fences (```):  weight 0.20  cap at 3 occurrences
          - Multi-step markers: weight 0.25  cap at 4 phrase matches (case-insensitive)
          - Reasoning keywords: weight 0.25  cap at 3 keyword matches (case-insensitive)
          - Vision/Multimodal:  weight 0.60  binary (deterministic: if image is present)

        Each signal is individually clamped to [0, max_weight] before summing.
        The total is clamped to [0.0, 1.0].

        Inspired by RouteLLM (Apache-2.0) — lm-sys/RouteLLM
        """
        if not messages:
            return 0.0

        # Extract all text content from OpenAI-format messages
        text_parts: List[str] = []
        has_image = False
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                # Vision / multimodal: detect images and collect text
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    # Deterministic check for image content
                    if block.get("type") in ("image_url", "image"):
                        has_image = True
                    text_parts.append(block.get("text", ""))
        raw_text = " ".join(text_parts)
        # Lowercase for case-insensitive matching; length uses raw (same char count)
        text = raw_text.lower()

        score = 0.0

        # Signal 1: character count — linear, cap at 10 000 chars (weight 0.30)
        score += min(len(raw_text) / 10_000, 1.0) * 0.30

        # Signal 2: code fence occurrences — cap at 3 (weight 0.20)
        code_fence_count = raw_text.count("```")
        score += min(code_fence_count / 3, 1.0) * 0.20

        # Signal 3: multi-step step-number markers — cap at 4 (weight 0.25)
        multi_step_patterns = ["step 1", "then,", "first,", "next,", "finally,"]
        marker_count = sum(1 for p in multi_step_patterns if p in text)
        score += min(marker_count / 4, 1.0) * 0.25

        # Signal 4: deep-reasoning keywords — cap at 3 (weight 0.25)
        reasoning_keywords = [
            "analyze", "compare", "evaluate", "explain why",
            "trade-off", "pros and cons",
        ]
        keyword_count = sum(1 for kw in reasoning_keywords if kw in text)
        score += min(keyword_count / 3, 1.0) * 0.25

        # Signal 5: Vision/Multimodal content (weight 0.60)
        # Deterministic: image presence implies high visual complexity
        if has_image:
            score += 0.60

        return min(1.0, score)

    def get_complexity_model(
        self, complexity: float, model_classes: Dict[str, Any]
    ) -> Optional[List[str]]:
        """Map a complexity score to MODEL_CLASSES tier candidates.

        Returns a list of candidate model names for the appropriate tier,
        or None when complexity is high (caller uses any available model).

        Tier boundaries:
          score < 0.35          → small tier  (MODEL_CLASSES["historical_small"])
          0.35 ≤ score ≤ 0.65  → medium tier (MODEL_CLASSES["historical_medium"])
          score > 0.65          → any tier    (returns None → caller selects freely)
        """
        if complexity < self.LOW_THRESHOLD:
            cls = model_classes.get("historical_small") or {}
            candidates = cls.get("candidates", [])
            return candidates if candidates else None
        # HIGH_THRESHOLD is inclusive: score == 0.65 is still medium
        if complexity <= self.HIGH_THRESHOLD:
            cls = model_classes.get("historical_medium") or {}
            candidates = cls.get("candidates", [])
            return candidates if candidates else None
        return None  # High complexity → caller uses any available model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
STRATEGIES = {
    "ROUND_ROBIN": RoundRobinStrategy,
    "RANDOM": RandomStrategy,
    "LEAST_LOADED": LeastLoadedStrategy,
    "P2C": PowerOfTwoChoicesStrategy,
    "POWER_OF_TWO": PowerOfTwoChoicesStrategy,
    "CONSISTENT_HASH": ConsistentHashingStrategy,
    "COMPLEXITY": ComplexityRoutingStrategy,
}

def get_routing_strategy(strategy_name: str, **kwargs) -> RoutingStrategy:
    strategy_class = STRATEGIES.get(strategy_name.upper())
    if not strategy_class:
        raise ValueError(f"Unknown routing strategy: {strategy_name}")

    # Pass configuration parameters for strategies that support them
    if strategy_name.upper() in ("P2C", "POWER_OF_TWO"):
        return strategy_class(
            alpha=kwargs.get("alpha", _config.P2C_ALPHA),
            penalty_weight=kwargs.get("penalty_weight", _config.P2C_PENALTY_WEIGHT),
            beta=kwargs.get("beta", _config.P2C_BETA),
        )
    elif strategy_name.upper() == "CONSISTENT_HASH":
        return strategy_class(virtual_nodes=kwargs.get("virtual_nodes", 150))
    else:
        return strategy_class()

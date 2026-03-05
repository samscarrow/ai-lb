"""Tests for ComplexityRoutingStrategy.

Covers:
  TC-1  Threshold constants: LOW=0.10, HIGH=0.25
  TC-2  Config-driven thresholds (COMPLEXITY_THRESHOLD_LOW / _HIGH)
  TC-3  score_prompt – empty / trivial prompt scores near zero (< LOW)
  TC-4  score_prompt – medium prompt (step-by-step) in [LOW, HIGH)
  TC-5  score_prompt – complex prompt (multi-signal) >= HIGH
  TC-6  classify – maps score ranges to correct tier names
  TC-7  select_node – no nodes returns None
  TC-8  select_node – no prompt delegates to fallback
  TC-9  select_node – tier selection routes to small-tier node for simple prompt
  TC-10 select_node – tier fallback: requested tier unavailable → adjacent tier
  TC-11 get_routing_strategy factory creates ComplexityRoutingStrategy
  TC-12 Scorer signal caps: single saturated signal cannot exceed its weight
"""

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_balancer.routing.strategies import ComplexityRoutingStrategy, get_routing_strategy
from load_balancer import config as lb_config


def run(coro):
    return asyncio.run(coro)


class FakeRedis:
    """Minimal in-memory Redis stub for unit tests."""

    def __init__(self):
        self.kv: dict = {}
        self.sets: dict = {"nodes:healthy": set()}

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, val):
        self.kv[key] = val


# ---------------------------------------------------------------------------
# TC-1: Threshold constants are LOW=0.10, HIGH=0.25
# ---------------------------------------------------------------------------

def test_threshold_constant_low():
    assert ComplexityRoutingStrategy.LOW == 0.10


def test_threshold_constant_high():
    assert ComplexityRoutingStrategy.HIGH == 0.25


# ---------------------------------------------------------------------------
# TC-2: Config-driven thresholds
# ---------------------------------------------------------------------------

def test_config_thresholds_applied(monkeypatch):
    monkeypatch.setattr(lb_config, "COMPLEXITY_THRESHOLD_LOW", 0.15)
    monkeypatch.setattr(lb_config, "COMPLEXITY_THRESHOLD_HIGH", 0.30)
    strategy = ComplexityRoutingStrategy()
    assert strategy._low == 0.15
    assert strategy._high == 0.30


def test_constructor_overrides_config(monkeypatch):
    monkeypatch.setattr(lb_config, "COMPLEXITY_THRESHOLD_LOW", 0.15)
    monkeypatch.setattr(lb_config, "COMPLEXITY_THRESHOLD_HIGH", 0.30)
    strategy = ComplexityRoutingStrategy(low=0.05, high=0.20)
    assert strategy._low == 0.05
    assert strategy._high == 0.20


# ---------------------------------------------------------------------------
# TC-3: score_prompt – trivial prompt scores below LOW (0.10)
# ---------------------------------------------------------------------------

def test_score_trivial_prompt_below_low():
    strategy = ComplexityRoutingStrategy()
    score = strategy.score_prompt("hi")
    assert score < strategy.LOW, f"Expected score < {strategy.LOW}, got {score}"


def test_score_empty_prompt_is_zero():
    strategy = ComplexityRoutingStrategy()
    assert strategy.score_prompt("") == 0.0


# ---------------------------------------------------------------------------
# TC-4: score_prompt – medium prompt lands in [LOW, HIGH)
# ---------------------------------------------------------------------------

def test_score_medium_prompt_in_range():
    strategy = ComplexityRoutingStrategy()
    # A step-by-step question with one depth indicator but no vocab/structure signals
    prompt = "Explain how a load balancer works, step by step."
    score = strategy.score_prompt(prompt)
    assert strategy.LOW <= score < strategy.HIGH, (
        f"Expected {strategy.LOW} <= score < {strategy.HIGH}, got {score}"
    )


# ---------------------------------------------------------------------------
# TC-5: score_prompt – complex prompt scores >= HIGH (0.25)
# ---------------------------------------------------------------------------

def test_score_complex_prompt_at_or_above_high():
    strategy = ComplexityRoutingStrategy()
    prompt = (
        "Design a distributed microservice architecture with Kubernetes that uses JWT "
        "authentication, database sharding, and a vector-based retrieval system. "
        "Step by step, compare the latency vs throughput tradeoffs, and analyze how "
        "the algorithm handles concurrency under high load. Include API design for "
        "OAuth integration and cache invalidation strategies.\n"
        "```python\n# example scaffold\n```"
    )
    score = strategy.score_prompt(prompt)
    assert score >= strategy.HIGH, (
        f"Expected score >= {strategy.HIGH}, got {score}"
    )


# ---------------------------------------------------------------------------
# TC-6: classify – correct tier for each score region
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected_tier", [
    (0.00, "small"),
    (0.05, "small"),
    (0.09, "small"),
    (0.10, "medium"),
    (0.15, "medium"),
    (0.24, "medium"),
    (0.25, "large"),
    (0.30, "large"),
    (0.50, "large"),
])
def test_classify_tier(score, expected_tier):
    strategy = ComplexityRoutingStrategy()
    assert strategy.classify(score) == expected_tier


# ---------------------------------------------------------------------------
# TC-7: select_node – empty node list returns None
# ---------------------------------------------------------------------------

def test_select_node_empty_nodes():
    strategy = ComplexityRoutingStrategy()
    result = run(strategy.select_node([], "m", FakeRedis(), prompt="hello"))
    assert result is None


# ---------------------------------------------------------------------------
# TC-8: select_node – no prompt delegates to fallback (returns a node)
# ---------------------------------------------------------------------------

def test_select_node_no_prompt_uses_fallback():
    strategy = ComplexityRoutingStrategy()
    nodes = ["node-a:8080", "node-b:8080"]
    result = run(strategy.select_node(nodes, "m", FakeRedis(), prompt=None))
    assert result in nodes


def test_select_node_empty_prompt_uses_fallback():
    strategy = ComplexityRoutingStrategy()
    nodes = ["node-a:8080"]
    result = run(strategy.select_node(nodes, "m", FakeRedis(), prompt=""))
    assert result in nodes


# ---------------------------------------------------------------------------
# TC-9: select_node – tier selection routes to small-tier node for simple prompt
# ---------------------------------------------------------------------------

def test_select_node_simple_prompt_routes_to_small_tier():
    """A trivial prompt should select nodes from the small tier."""
    tiers = {
        "small": ["small-model"],
        "medium": ["medium-model"],
        "large": ["large-model"],
    }
    strategy = ComplexityRoutingStrategy(model_tiers=tiers)
    nodes = ["small-model:11434", "medium-model:11434", "large-model:11434"]
    result = run(strategy.select_node(nodes, "m", FakeRedis(), prompt="hi"))
    assert "small-model" in result


def test_select_node_complex_prompt_routes_to_large_tier():
    """A complex, multi-signal prompt should select nodes from the large tier."""
    tiers = {
        "small": ["small-model"],
        "medium": ["medium-model"],
        "large": ["large-model"],
    }
    strategy = ComplexityRoutingStrategy(model_tiers=tiers)
    nodes = ["small-model:11434", "medium-model:11434", "large-model:11434"]
    prompt = (
        "Design a distributed microservice architecture with Kubernetes that uses JWT "
        "authentication, database sharding, and a vector-based retrieval system. "
        "Step by step, compare the latency vs throughput tradeoffs, and analyze how "
        "the algorithm handles concurrency under high load. Include API design for "
        "OAuth integration and cache invalidation strategies.\n"
        "```python\n# example\n```"
    )
    result = run(strategy.select_node(nodes, "m", FakeRedis(), prompt=prompt))
    assert "large-model" in result


# ---------------------------------------------------------------------------
# TC-10: select_node – tier fallback when preferred tier is unavailable
# ---------------------------------------------------------------------------

def test_select_node_tier_fallback_to_adjacent():
    """When the preferred tier has no matching nodes, fall back to an adjacent tier."""
    tiers = {
        "small": ["small-model"],
        "medium": ["medium-model"],
        "large": ["large-model"],
    }
    strategy = ComplexityRoutingStrategy(model_tiers=tiers)
    # Only medium and large nodes are available
    nodes = ["medium-model:11434", "large-model:11434"]
    # Simple prompt → small tier requested → no small nodes → should fall back
    result = run(strategy.select_node(nodes, "m", FakeRedis(), prompt="hi"))
    assert result in nodes  # must pick one of the available nodes


def test_select_node_no_tier_nodes_uses_all_nodes():
    """When no tier matches any node, all available nodes are candidates."""
    tiers = {
        "small": ["tiny-model"],
        "medium": ["mid-model"],
        "large": ["big-model"],
    }
    strategy = ComplexityRoutingStrategy(model_tiers=tiers)
    # Nodes don't match any tier candidate
    nodes = ["unknown-model:11434"]
    result = run(strategy.select_node(nodes, "m", FakeRedis(), prompt="hi"))
    assert result in nodes


# ---------------------------------------------------------------------------
# TC-11: get_routing_strategy factory
# ---------------------------------------------------------------------------

def test_get_routing_strategy_factory_creates_complexity():
    strategy = get_routing_strategy("COMPLEXITY")
    assert isinstance(strategy, ComplexityRoutingStrategy)


def test_get_routing_strategy_factory_with_threshold_overrides():
    strategy = get_routing_strategy("COMPLEXITY", low=0.05, high=0.20)
    assert strategy._low == 0.05
    assert strategy._high == 0.20


# ---------------------------------------------------------------------------
# TC-12: Signal caps — a single saturated signal cannot exceed its weight
# ---------------------------------------------------------------------------

def test_single_signal_length_capped():
    """An extremely long prompt saturates the length signal but cannot exceed its weight."""
    strategy = ComplexityRoutingStrategy()
    very_long = "word " * 5000  # Far beyond the high length threshold
    score = strategy.score_prompt(very_long)
    # Only the length signal fires; the others are zero → score == length weight
    assert score <= strategy._SIGNAL_LENGTH_WEIGHT + 1e-9


def test_score_bounded_above_by_one():
    """Score must never exceed 1.0 regardless of input."""
    strategy = ComplexityRoutingStrategy()
    adversarial = (
        "```python\narchitecture algorithm distributed concurrency latency throughput "
        "cache database kubernetes microservice api oauth jwt encryption inference "
        "fine-tuning gradient transformer embedding vector retrieval\n``` " * 10
        + "step by step compare versus vs. explain why pros and cons tradeoff "
        "what if how does analyze " * 5
        + "x" * 5000
    )
    score = strategy.score_prompt(adversarial)
    assert score <= 1.0

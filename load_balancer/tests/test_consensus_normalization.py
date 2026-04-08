"""Unit tests for consensus output normalization.

These tests avoid FastAPI dependencies and only exercise the pure consensus logic.
"""

import sys
from pathlib import Path

# Ensure package path for local src
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_balancer.execution import BackendResult, compute_consensus


def _resp(content: str):
    return {
        "id": "x",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def test_thinking_tags_are_stripped_for_hash_and_similarity():
    r1 = BackendResult(
        backend="cloud:openai",
        success=True,
        response_body=_resp("<thinking>foo</thinking>\n4"),
        latency_ms=10,
    )
    r2 = BackendResult(
        backend="localhost:11434",
        success=True,
        response_body=_resp("4"),
        latency_ms=5,
    )

    c = compute_consensus([r1, r2], similarity_threshold=0.9)

    assert c.agreement_count == 2
    assert c.disagreement is False


def test_fenced_json_is_canonicalized_and_matches():
    a = """```json
    {"b": 2, "a": 1}
    ```"""
    b = """{\n  "a": 1,\n  "b": 2\n}"""

    r1 = BackendResult(backend="b1", success=True, response_body=_resp(a), latency_ms=10)
    r2 = BackendResult(backend="b2", success=True, response_body=_resp(b), latency_ms=5)

    c = compute_consensus([r1, r2], similarity_threshold=0.9)

    assert c.agreement_count == 2
    assert c.disagreement is False


def test_invalid_json_does_not_match_valid_json():
    valid = '{"a": 1}'
    invalid = '{"a": 1,}'

    r1 = BackendResult(backend="b1", success=True, response_body=_resp(valid), latency_ms=10)
    r2 = BackendResult(backend="b2", success=True, response_body=_resp(invalid), latency_ms=5)

    c = compute_consensus([r1, r2], similarity_threshold=0.9)

    assert c.agreement_count == 1
    assert c.disagreement is True

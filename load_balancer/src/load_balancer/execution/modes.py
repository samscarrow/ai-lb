"""Multi-backend execution modes and consensus algorithms."""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for multi-backend requests."""
    RACE = "race"           # N-way hedging, first response wins
    ALL = "all"             # Collect all responses in parallel
    SEQUENCE = "sequence"   # Serial chain through backends
    CONSENSUS = "consensus" # Majority vote - 2 of 3 agree wins


@dataclass
class ExecutionConfig:
    """Configuration for a multi-backend execution request."""
    mode: ExecutionMode
    target_backends: Optional[List[str]] = None  # Specific backends to use (aliases or host:port)
    max_backends: int = 3
    timeout_secs: float = 60.0


@dataclass
class BackendResult:
    """Result from a single backend execution."""
    backend: str                        # host:port of the backend
    alias: Optional[str] = None         # Human-friendly alias if configured
    success: bool = False
    status_code: int = 0
    response_body: Optional[Dict] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    first_byte_ms: float = 0.0          # Time to first byte for streaming


@dataclass
class ConsensusResult:
    """Result of consensus computation across multiple backends."""
    winner: BackendResult               # The majority-agreed response
    all_responses: List[BackendResult] = field(default_factory=list)
    agreement_count: int = 0            # How many backends agreed (2 or 3)
    disagreement: bool = False          # True if not unanimous
    comparison_type: str = "unknown"    # "tool_calls" | "text" | "hash"


def _normalize_tool_calls(tool_calls: List[Dict]) -> str:
    """Normalize tool calls for comparison by sorting and normalizing JSON."""
    if not tool_calls:
        return ""
    normalized = []
    for tc in tool_calls:
        call = {
            "name": tc.get("function", {}).get("name", ""),
            "arguments": tc.get("function", {}).get("arguments", "")
        }
        # Parse and re-serialize arguments to normalize JSON formatting
        try:
            args = json.loads(call["arguments"]) if call["arguments"] else {}
            call["arguments"] = json.dumps(args, sort_keys=True, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            pass
        normalized.append(call)
    # Sort by name for consistent ordering
    normalized.sort(key=lambda x: x["name"])
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _extract_tool_calls(response: Dict) -> Optional[List[Dict]]:
    """Extract tool_calls from an OpenAI-format response."""
    choices = response.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return tool_calls
    # Also check for function_call (older format)
    function_call = message.get("function_call")
    if function_call:
        return [{"function": function_call}]
    return None


def _extract_text_content(response: Dict) -> str:
    """Extract text content from an OpenAI-format response."""
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content", "") or ""


def _compute_text_similarity(text1: str, text2: str) -> float:
    """Compute similarity ratio between two text strings using SequenceMatcher."""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    # Normalize whitespace
    t1 = " ".join(text1.split())
    t2 = " ".join(text2.split())
    return SequenceMatcher(None, t1, t2).ratio()


def _content_hash(response: Dict) -> str:
    """Compute a hash of the response content for exact comparison."""
    text = _extract_text_content(response)
    tool_calls = _extract_tool_calls(response)
    content = {
        "text": " ".join(text.split()) if text else "",
        "tool_calls": _normalize_tool_calls(tool_calls) if tool_calls else ""
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def compute_consensus(
    results: List[BackendResult],
    similarity_threshold: float = 0.9
) -> ConsensusResult:
    """
    Compute consensus from multiple backend results.

    For tool calls: Compare normalized action name + parameters.
    For text: Compare content hash, fall back to similarity ratio.

    Args:
        results: List of BackendResult from different backends
        similarity_threshold: Minimum similarity for text agreement (0.0-1.0)

    Returns:
        ConsensusResult with winner and agreement info
    """
    successful = [r for r in results if r.success and r.response_body]

    if not successful:
        # All failed - return first result with error info
        return ConsensusResult(
            winner=results[0] if results else BackendResult(backend="unknown", error="No results"),
            all_responses=results,
            agreement_count=0,
            disagreement=True,
            comparison_type="none"
        )

    if len(successful) == 1:
        # Only one success - it wins by default
        return ConsensusResult(
            winner=successful[0],
            all_responses=results,
            agreement_count=1,
            disagreement=False,
            comparison_type="single"
        )

    # Check if responses have tool calls
    tool_call_responses = []
    for r in successful:
        tc = _extract_tool_calls(r.response_body)
        if tc:
            tool_call_responses.append((r, _normalize_tool_calls(tc)))

    # Prefer tool call comparison if most responses have tool calls
    if len(tool_call_responses) >= len(successful) // 2 + 1:
        return _consensus_by_tool_calls(successful, tool_call_responses, results, similarity_threshold)

    # Fall back to text comparison
    return _consensus_by_text(successful, results, similarity_threshold)


def _consensus_by_tool_calls(
    successful: List[BackendResult],
    tool_call_data: List[Tuple[BackendResult, str]],
    all_results: List[BackendResult],
    similarity_threshold: float
) -> ConsensusResult:
    """Compute consensus based on tool call comparison."""
    # Group by normalized tool calls
    groups: Dict[str, List[BackendResult]] = {}
    for result, normalized in tool_call_data:
        if normalized not in groups:
            groups[normalized] = []
        groups[normalized].append(result)

    # Find largest group
    largest_group = max(groups.values(), key=len)
    winner = min(largest_group, key=lambda r: r.latency_ms)  # Pick fastest in consensus group

    return ConsensusResult(
        winner=winner,
        all_responses=all_results,
        agreement_count=len(largest_group),
        disagreement=len(largest_group) < len(successful),
        comparison_type="tool_calls"
    )


def _consensus_by_text(
    successful: List[BackendResult],
    all_results: List[BackendResult],
    similarity_threshold: float
) -> ConsensusResult:
    """Compute consensus based on text content comparison."""
    # First try exact hash match
    hashes: Dict[str, List[BackendResult]] = {}
    for r in successful:
        h = _content_hash(r.response_body)
        if h not in hashes:
            hashes[h] = []
        hashes[h].append(r)

    # Check if we have hash-based majority
    largest_hash_group = max(hashes.values(), key=len)
    if len(largest_hash_group) > len(successful) // 2:
        winner = min(largest_hash_group, key=lambda r: r.latency_ms)
        return ConsensusResult(
            winner=winner,
            all_responses=all_results,
            agreement_count=len(largest_hash_group),
            disagreement=len(largest_hash_group) < len(successful),
            comparison_type="hash"
        )

    # Fall back to similarity-based grouping
    texts = [(r, _extract_text_content(r.response_body)) for r in successful]

    # Build similarity matrix and group by threshold
    groups: List[List[BackendResult]] = []
    used = set()

    for i, (r1, t1) in enumerate(texts):
        if i in used:
            continue
        group = [r1]
        used.add(i)
        for j, (r2, t2) in enumerate(texts):
            if j in used:
                continue
            if _compute_text_similarity(t1, t2) >= similarity_threshold:
                group.append(r2)
                used.add(j)
        groups.append(group)

    largest_group = max(groups, key=len)
    winner = min(largest_group, key=lambda r: r.latency_ms)

    return ConsensusResult(
        winner=winner,
        all_responses=all_results,
        agreement_count=len(largest_group),
        disagreement=len(largest_group) < len(successful),
        comparison_type="text"
    )


class ExecutionEngine:
    """Engine for executing requests across multiple backends."""

    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold

    async def execute_race(
        self,
        backends: List[str],
        request_fn: Callable[[str], Awaitable[BackendResult]],
        timeout: float = 60.0
    ) -> BackendResult:
        """
        Race N backends, return first successful response.

        Args:
            backends: List of backend host:port strings
            request_fn: Async function that takes backend and returns BackendResult
            timeout: Overall timeout in seconds

        Returns:
            First successful BackendResult, or first error if all fail
        """
        if not backends:
            return BackendResult(backend="none", error="No backends provided")

        tasks = {
            asyncio.create_task(request_fn(b)): b
            for b in backends
        }

        winner = None
        errors = []

        try:
            while tasks and not winner:
                done, pending = await asyncio.wait(
                    tasks.keys(),
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )

                if not done:
                    # Timeout
                    break

                for task in done:
                    try:
                        result = task.result()
                        if result.success:
                            winner = result
                            break
                        else:
                            errors.append(result)
                    except Exception as e:
                        backend = tasks.get(task, "unknown")
                        errors.append(BackendResult(
                            backend=backend,
                            error=str(e)
                        ))
                    del tasks[task]

        finally:
            # Cancel remaining tasks
            for task in tasks:
                task.cancel()

        if winner:
            return winner
        if errors:
            return errors[0]
        return BackendResult(backend="timeout", error="All backends timed out")

    async def execute_all(
        self,
        backends: List[str],
        request_fn: Callable[[str], Awaitable[BackendResult]],
        timeout: float = 60.0
    ) -> List[BackendResult]:
        """
        Execute request on all backends in parallel, collect all responses.

        Args:
            backends: List of backend host:port strings
            request_fn: Async function that takes backend and returns BackendResult
            timeout: Overall timeout in seconds

        Returns:
            List of BackendResult from all backends
        """
        if not backends:
            return []

        async def wrapped_request(backend: str) -> BackendResult:
            try:
                return await asyncio.wait_for(request_fn(backend), timeout=timeout)
            except asyncio.TimeoutError:
                return BackendResult(backend=backend, error="Timeout")
            except Exception as e:
                return BackendResult(backend=backend, error=str(e))

        tasks = [wrapped_request(b) for b in backends]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def execute_sequence(
        self,
        backends: List[str],
        request_fn: Callable[[str], Awaitable[BackendResult]],
        timeout: float = 60.0,
        stop_on_success: bool = False
    ) -> List[BackendResult]:
        """
        Execute request on backends sequentially.

        Args:
            backends: List of backend host:port strings (in order)
            request_fn: Async function that takes backend and returns BackendResult
            timeout: Per-backend timeout in seconds
            stop_on_success: If True, stop after first successful response

        Returns:
            List of BackendResult from attempted backends
        """
        results = []
        for backend in backends:
            try:
                result = await asyncio.wait_for(request_fn(backend), timeout=timeout)
                results.append(result)
                if stop_on_success and result.success:
                    break
            except asyncio.TimeoutError:
                results.append(BackendResult(backend=backend, error="Timeout"))
            except Exception as e:
                results.append(BackendResult(backend=backend, error=str(e)))
        return results

    async def execute_consensus(
        self,
        backends: List[str],
        request_fn: Callable[[str], Awaitable[BackendResult]],
        timeout: float = 60.0
    ) -> ConsensusResult:
        """
        Execute request on all backends and compute consensus.

        Args:
            backends: List of backend host:port strings
            request_fn: Async function that takes backend and returns BackendResult
            timeout: Overall timeout in seconds

        Returns:
            ConsensusResult with winner and agreement info
        """
        results = await self.execute_all(backends, request_fn, timeout)
        return compute_consensus(results, self.similarity_threshold)

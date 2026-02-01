"""Multi-backend execution engine for parallel/sequential LLM calls."""

from .modes import (
    ExecutionMode,
    ExecutionConfig,
    BackendResult,
    ConsensusResult,
    ExecutionEngine,
    compute_consensus,
)

__all__ = [
    "ExecutionMode",
    "ExecutionConfig",
    "BackendResult",
    "ConsensusResult",
    "ExecutionEngine",
    "compute_consensus",
]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    id: str
    description: str
    depends_on: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    tasks: List[Task]


@dataclass
class ToolCallRecord:
    tool: str
    arguments: Dict[str, Any]
    output: str
    error: Optional[str] = None


@dataclass
class LbCallMetadata:
    request_id: Optional[str]
    routed_node: Optional[str]
    selected_model: Optional[str]
    attempts: Optional[str]
    failover_count: Optional[str]
    fallback_model: Optional[str]
    status_code: int


@dataclass
class TaskResult:
    task_id: str
    output: str
    success: bool
    lb_calls: List[LbCallMetadata] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class OrchestrationResult:
    plan: Plan
    results: Dict[str, TaskResult]
    state: Dict[str, Any]

from __future__ import annotations

import argparse
import json
import sys
import uuid

from .orchestrator import Orchestrator
from .types import Task
from .worker import Worker, task_result_as_dict


def _parse_json(value: str) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(prog="hybrid-agent-client")
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker_parser = subparsers.add_parser("worker", help="Run a single Worker task")
    worker_parser.add_argument("--task", required=True, help="Task description")
    worker_parser.add_argument("--task-id", default=None, help="Optional task id")
    worker_parser.add_argument("--payload", default="", help="JSON payload for the task")

    orch_parser = subparsers.add_parser("orchestrator", help="Run Orchestrator end-to-end")
    orch_parser.add_argument("--request", required=True, help="User request")

    args = parser.parse_args()

    if args.command == "worker":
        task_id = args.task_id or f"task-{uuid.uuid4().hex[:8]}"
        payload = _parse_json(args.payload)
        worker = Worker()
        result = worker.run(Task(id=task_id, description=args.task, payload=payload))
        print(json.dumps(task_result_as_dict(result), indent=2))
        return 0

    if args.command == "orchestrator":
        orchestrator = Orchestrator()
        result = orchestrator.run(args.request)
        out = {
            "plan": [t.__dict__ for t in result.plan.tasks],
            "results": {task_id: task_result_as_dict(task_result) for task_id, task_result in result.results.items()},
            "state": result.state,
        }
        print(json.dumps(out, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

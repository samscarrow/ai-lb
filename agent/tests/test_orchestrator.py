from agent.orchestrator import Orchestrator, SimpleVerifier
from agent.types import Plan, Task, TaskResult


class FixedPlanner:
    def plan(self, request: str) -> Plan:
        return Plan(
            tasks=[
                Task(id="t1", description="first"),
                Task(id="t2", description="second", depends_on=["t1"]),
            ]
        )


class FakeWorker:
    def run(self, task: Task) -> TaskResult:
        return TaskResult(task_id=task.id, output=f"done:{task.id}", success=True)


def test_orchestrator_runs_dependency_order() -> None:
    orchestrator = Orchestrator(worker=FakeWorker(), planner=FixedPlanner(), verifier=SimpleVerifier())
    result = orchestrator.run("do the thing")
    assert list(result.results.keys()) == ["t1", "t2"]
    assert result.results["t2"].output == "done:t2"

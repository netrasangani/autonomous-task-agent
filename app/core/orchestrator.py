from typing import Any
import uuid

from app.core.state import AgentState
from app.agents.manager import manager_decide


class TaskOrchestrator:
    """
    Controls execution of an autonomous task.

    The orchestrator maintains state and repeatedly gives control
    to the Manager Agent. The Manager decides what should happen
    next; the orchestrator does not define the workflow.
    """

    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations

    def run(
        self,
        goal: str,
        file_path: str | None = None,
    ) -> dict[str, Any]:

        request_id = str(uuid.uuid4())

        state = AgentState(
            goal=goal,
            request_id=request_id,
        )

        final_result = None

        for iteration in range(1, self.max_iterations + 1):

            try:
                final_result = manager_decide(
                    state=state,
                    file_path=file_path,
                )

                if state.completed:
                    break

            except Exception as exc:
                state.history.append(
                    {
                        "iteration": iteration,
                        "stage": "error",
                        "error": str(exc),
                    }
                )

                final_result = f"Task execution failed: {exc}"
                break

        else:
            final_result = (
                "Task execution stopped because the maximum "
                f"iteration limit of {self.max_iterations} was reached."
            )

        return {
            "request_id": request_id,
            "goal": goal,
            "file_path": file_path,
            "completed": state.completed,
            "iterations": iteration,
            "history": state.history,
            "results": state.results,
            "final_result": final_result,
        }


def run_task(
    goal: str,
    file_path: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function for executing an autonomous task.
    """

    orchestrator = TaskOrchestrator()

    return orchestrator.run(
        goal=goal,
        file_path=file_path,
    )
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    goal: str
    request_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False

    def add_action(self, agent: str, action: str, result: Any):
        self.history.append({
            "agent": agent,
            "action": action,
        })

        self.results.append({
            "agent": agent,
            "action": action,
            "result": result,
        })
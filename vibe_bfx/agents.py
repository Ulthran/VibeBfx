from __future__ import annotations

"""Core agent implementations used by the task graph."""

from typing import Any, Callable, Dict, List, TypedDict


class Step(TypedDict, total=False):
    """Representation of a single unit of work in a plan."""

    description: str
    tool: Callable[..., Any]
    inputs: Dict[str, Any]
    params: Dict[str, Any]
    notes: str


class Runner:
    """Execute the command defined by a plan step."""

    def run(self, step: Step) -> Any:
        tool = step["tool"]
        inputs = step.get("inputs", {})
        params = step.get("params", {})
        return tool(**inputs, **params)


class Analyzer:
    """Analyze step results and produce a report string."""

    def analyze(self, step: Step, result: Any) -> str:
        desc = step.get("description", "Step")
        return f"{desc}: {result}"


class Planner:
    """Create and iteratively refine a plan for a task."""

    def __init__(self, task: "Task"):
        self.task = task

    def plan(
        self,
        prompt: str,
        *,
        tool: Callable[..., Any],
        inputs: Dict[str, Any],
        params: Dict[str, Any] | None = None,
    ) -> List[Step]:
        params = params or {}
        return [
            {
                "description": prompt,
                "tool": tool,
                "inputs": inputs,
                "params": params,
                "notes": "",
            }
        ]

    def replan(self, plan: List[Step], reports: List[str]) -> List[Step]:
        """Re-evaluate the plan given completed step reports.

        The default implementation simply returns the existing plan unchanged.
        """

        return plan


__all__ = ["Step", "Runner", "Analyzer", "Planner"]


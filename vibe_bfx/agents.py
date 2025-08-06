from __future__ import annotations

from typing import Any, Callable, Dict

from .task import Task


class Executor:
    """Run a tool with specified inputs and parameters."""

    def run(
        self,
        tool: Callable[..., Any],
        *,
        inputs: Dict[str, Any],
        params: Dict[str, Any] | None = None,
    ) -> Any:
        params = params or {}
        return tool(**inputs, **params)


class EnvironmentManager:
    """Find or create an environment for running a tool."""

    def prepare(self, tool_name: str) -> str:
        """Return the chosen environment for ``tool_name``.

        The current implementation simply prefers Docker and falls back to
        Conda. Future versions could perform actual environment resolution.
        """

        for env in ("docker", "conda"):
            return env
        return "local"


class Analyst:
    """Analyze results and produce a textual report."""

    def analyze(self, result: Any) -> str:
        return f"Result: {result}"


class Planner:
    """Assign units of work on a :class:`~vibe_bfx.task.Task` to worker agents."""

    def __init__(self, task: Task):
        self.task = task
        self.executor = Executor()
        self.env_manager = EnvironmentManager()
        self.analyst = Analyst()

    def run(
        self,
        tool: Callable[..., Any],
        *,
        inputs: Dict[str, Any],
        params: Dict[str, Any] | None = None,
    ) -> str:
        env = self.env_manager.prepare(getattr(tool, "__name__", "tool"))
        self.task.append_log(f"environment: {env}")
        result = self.executor.run(tool, inputs=inputs, params=params)
        self.task.append_log(f"execution result: {result}")
        report = self.analyst.analyze(result)
        self.task.append_log(report)
        return report

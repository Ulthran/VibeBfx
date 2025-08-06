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
        tool_name = getattr(tool, "__name__", "tool")
        with self.task.log_context("planner") as logger:
            logger.info("plan: run %s with inputs %s", tool_name, inputs)
            env = self.task.run_agent(
                "environment",
                self.env_manager.prepare,
                tool_name,
                result_label="environment",
            )
            logger.info("environment: %s", env)
            result = self.task.run_agent(
                "executor",
                self.executor.run,
                tool,
                inputs=inputs,
                params=params,
                result_label="execution result",
            )
            logger.info("execution result: %s", result)
            report = self.task.run_agent(
                "analyst",
                self.analyst.analyze,
                result,
                result_label="analysis",
            )
            logger.info("analysis: %s", report)
        return report

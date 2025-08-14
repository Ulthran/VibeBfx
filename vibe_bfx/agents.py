from __future__ import annotations

from typing import Any, Callable, Dict

import cloudpickle

from .celery_app import app, execute_tool


class Runner:
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
    """Prepare the execution environment for a tool.

    In a real deployment this might provision containers or other
    resources.  For now it simply returns a placeholder string so that
    higher level orchestration can proceed.
    """

    def prepare(self, tool_name: str) -> str:
        return "docker"


class Executor:
    """Execute callables via Celery.

    The :mod:`celery` app is configured for eager execution during tests
    so calls block until completion, but in production the same code can
    dispatch work to remote workers.
    """

    def __init__(self) -> None:
        self.app = app

    def run(
        self,
        tool: Callable[..., Any],
        *,
        inputs: Dict[str, Any],
        params: Dict[str, Any] | None = None,
    ) -> Any:
        params = params or {}
        payload = cloudpickle.dumps(tool)
        result = execute_tool.delay(payload, inputs, params)
        return result.get()


class Analyst:
    """Analyze results and produce a textual report."""

    def analyze(self, result: Any) -> str:
        return f"Result: {result}"


class Planner:
    """Assign units of work on a :class:`~vibe_bfx.task.Task` to worker agents."""

    def __init__(self, task: "Task"):
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

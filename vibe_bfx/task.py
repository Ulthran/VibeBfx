from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

import logging
from datetime import datetime
from contextlib import contextmanager

from langchain.schema import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph


class ChatState(TypedDict):
    messages: List[BaseMessage]


class PlanState(TypedDict):
    prompt: str
    plan: List[dict] | None
    step_index: int
    current_step: dict | None
    last_result: Any | None
    reports: List[str]


class Task:
    """Represents a unit of work inside a project."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.chat_file = self.path / "chat.txt"
        self.log_file = self.path / "log.txt"
        self.logs_dir = self.path / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for f in (self.chat_file, self.log_file):
            f.touch(exist_ok=True)

    def append_chat(self, role: str, message: str) -> None:
        with self.chat_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{role}: {message}\n")

    def append_log(self, entry: str) -> None:
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(entry + "\n")

    def append_log_reference(self, node: str, log_path: Path, timestamp: str) -> None:
        rel = log_path.relative_to(self.path)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} {node}: {rel.as_posix()}\n")

    @contextmanager
    def log_context(self, node: str):
        """Context manager yielding a logger writing to the task's logs directory."""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = self.logs_dir / f"{ts}_{node}.log"

        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)

        logger = logging.getLogger(f"{node}-{ts}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(handler)

        try:
            yield logger
        finally:
            logger.removeHandler(handler)
            handler.close()
            self.append_log_reference(node, log_path, ts)

    def run_agent(
        self,
        node: str,
        func: Callable[..., Any],
        *args: Any,
        result_label: str = "result",
        **kwargs: Any,
    ) -> Any:
        """Run ``func`` with logging and return its result."""
        with self.log_context(node) as logger:
            if args or kwargs:
                logger.info("inputs: args=%s kwargs=%s", args, kwargs)
            result = func(*args, **kwargs)
            logger.info(f"{result_label}: %s", result)
        return result

    def run(
        self,
        prompt: str,
        tool: Callable[..., Any],
        *,
        inputs: Dict[str, Any],
        params: Dict[str, Any] | None = None,
    ) -> str:
        """Run a planner-driven task using a LangGraph workflow."""

        from .agents import Analyzer, Planner, Runner

        self.append_chat("user", prompt)

        planner = Planner(self)
        runner = Runner()
        analyzer = Analyzer()

        def plan_node(state: PlanState) -> PlanState:
            if state["plan"] is None:
                plan = self.run_agent(
                    "planner",
                    planner.plan,
                    state["prompt"],
                    tool=tool,
                    inputs=inputs,
                    params=params,
                    result_label="plan",
                )
            else:
                plan = self.run_agent(
                    "planner",
                    planner.replan,
                    state["plan"],
                    state["reports"],
                    result_label="plan",
                )
            if state["step_index"] >= len(plan):
                return {**state, "plan": plan, "current_step": None}
            step = plan[state["step_index"]]
            return {**state, "plan": plan, "current_step": step}

        def runner_node(state: PlanState) -> PlanState:
            result = self.run_agent(
                "runner", runner.run, state["current_step"], result_label="result"
            )
            return {**state, "last_result": result}

        def analyzer_node(state: PlanState) -> PlanState:
            report = self.run_agent(
                "analyzer",
                analyzer.analyze,
                state["current_step"],
                state["last_result"],
                result_label="report",
            )
            reports = state["reports"] + [report]
            return {**state, "reports": reports, "step_index": state["step_index"] + 1}

        def route_from_planner(state: PlanState) -> str:
            return "runner" if state.get("current_step") else END

        graph = StateGraph(PlanState)
        graph.add_node("planner", plan_node)
        graph.add_node("runner", runner_node)
        graph.add_node("analyzer", analyzer_node)
        graph.add_conditional_edges("planner", route_from_planner)
        graph.add_edge("runner", "analyzer")
        graph.add_edge("analyzer", "planner")
        graph.set_entry_point("planner")
        app = graph.compile()

        final_state = app.invoke(
            {
                "prompt": prompt,
                "plan": None,
                "step_index": 0,
                "current_step": None,
                "last_result": None,
                "reports": [],
            }
        )

        report = "\n".join(final_state["reports"])
        self.append_chat("assistant", report)
        return report

    def run_chat(self, prompt: str, model: Optional[Any] = None) -> str:
        """Run a single chat turn using LangChain and LangGraph.

        Parameters
        ----------
        prompt:
            The user prompt to send to the model.
        model:
            Optional LangChain chat model. If ``None`` a default
            :class:`langchain_openai.ChatOpenAI` model is used.
        """

        if model is None:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )

        def call_model(state: ChatState) -> ChatState:
            response = self.run_agent(
                "model",
                model.invoke,
                state["messages"],
                result_label="response",
            )
            return {"messages": state["messages"] + [response]}

        graph = StateGraph(ChatState)
        graph.add_node("model", call_model)
        graph.add_edge("model", END)
        graph.set_entry_point("model")
        app = graph.compile()

        result = app.invoke({"messages": [HumanMessage(content=prompt)]})

        response = result["messages"][-1].content

        self.append_chat("user", prompt)
        self.append_chat("assistant", response)

        return response

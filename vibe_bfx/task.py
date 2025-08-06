from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

import logging
from datetime import datetime
from contextlib import contextmanager

from langchain.schema import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langchain_community.chat_models import ChatOpenAI


class ChatState(TypedDict):
    messages: List[BaseMessage]


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
        """Run a planner-driven task using ``tool`` and record chat/logs."""
        from .agents import Planner  # local import to avoid circular dependency

        self.append_chat("user", prompt)
        planner = Planner(self)
        report = planner.run(tool=tool, inputs=inputs, params=params)
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
            :class:`langchain_community.chat_models.ChatOpenAI` model is used.
        """

        if model is None:
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

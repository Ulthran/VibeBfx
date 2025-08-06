from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, TypedDict

import logging
from datetime import datetime

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
            node = "model"
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_path = self.logs_dir / f"{ts}_{node}.log"

            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            handler.setFormatter(formatter)

            logger = logging.getLogger(node)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            logger.addHandler(handler)

            try:
                logger.info("Prompt: %s", state["messages"][-1].content)
                response = model.invoke(state["messages"])
                logger.info("Response: %s", response.content)
            finally:
                logger.removeHandler(handler)
                handler.close()
                self.append_log_reference(node, log_path, ts)

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

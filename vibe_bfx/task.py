from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, TypedDict

import io
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout

from langchain.schema import BaseMessage, HumanMessage


class Task:
    """Represents a unit of work inside a project."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.chat_file = self.path / "chat.txt"
        self.log_file = self.path / "log.txt"
        for f in (self.chat_file, self.log_file):
            f.touch(exist_ok=True)

    def append_chat(self, role: str, message: str) -> None:
        with self.chat_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{role}: {message}\n")

    def append_log(self, entry: str) -> None:
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(entry + "\n")

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

        class _Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data: str) -> int:
                for s in self.streams:
                    s.write(data)
                return len(data)

            def flush(self) -> None:  # pragma: no cover - passthrough
                for s in self.streams:
                    s.flush()

        buffer = io.StringIO()
        tee_out = _Tee(sys.stdout, buffer)
        tee_err = _Tee(sys.stderr, buffer)

        with warnings.catch_warnings(), redirect_stdout(tee_out), redirect_stderr(tee_err):
            warnings.simplefilter("default")

            if model is None:
                from langchain_openai import ChatOpenAI
                model = ChatOpenAI()

            from langgraph.graph import StateGraph, END

            class ChatState(TypedDict):
                messages: List[BaseMessage]

            def call_model(state: ChatState) -> ChatState:
                response = model.invoke(state["messages"])
                return {"messages": state["messages"] + [response]}

            graph = StateGraph(ChatState)
            graph.add_node("model", call_model)
            graph.add_edge("model", END)
            graph.set_entry_point("model")
            app = graph.compile()

            result = app.invoke({"messages": [HumanMessage(content=prompt)]})

        response = result["messages"][-1].content

        captured = buffer.getvalue().strip()
        if captured:
            self.append_log(captured)

        self.append_chat("user", prompt)
        self.append_chat("assistant", response)
        self.append_log(f"Prompt: {prompt}\nResponse: {response}\n")

        return response

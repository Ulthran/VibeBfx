from __future__ import annotations

from pathlib import Path
from typing import List, TypedDict

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
            :class:`langchain.chat_models.ChatOpenAI` model is used.
        """

        if model is None:
            from langchain.chat_models import ChatOpenAI
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

        self.append_chat("user", prompt)
        self.append_chat("assistant", response)
        self.append_log(f"Prompt: {prompt}\nResponse: {response}\n")

        return response

import subprocess
from celery import Celery
from langchain.schema import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from pathlib import Path
from typing import List, Sequence, TypedDict


class ChatState(TypedDict):
    messages: List[BaseMessage]


# The Celery application used throughout the project.  It defaults to an in
# memory broker/back end and eager execution so that unit tests do not require
# a separate worker process.  In production these settings can be overridden
# by environment variables or a configuration file.
app = Celery(broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
app.conf.update(
    task_always_eager=True,
    task_store_eager_result=True,
    task_serializer="pickle",
    result_serializer="pickle",
    accept_content=["pickle"],
)


@app.task
def do_work(prompt: str):
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        timeout=None,
        max_retries=2,
    )

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
    fp = Path("/home/ctbus/Penn/VibeBfx/out.txt")
    fp.write_text(response, encoding="utf-8")

    return response


@app.task()
def execute_tool(command: Sequence[str]) -> str:
    """Run ``command`` via :mod:`subprocess` and return its stdout."""

    completed = subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()

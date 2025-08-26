from langchain.schema import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pathlib import Path
from prefect import flow, task
from typing import List, TypedDict
from vibe_bfx.agents import Planner, Reporter, Runner


class ChatState(TypedDict):
    messages: List[BaseMessage]


planner = Planner()


def call_planner(state: ChatState) -> ChatState:
    response = planner.make_plan(state["messages"][-1])
    return {"messages": state["messages"] + [response]}


runner = Runner()


def call_runner(state: ChatState) -> ChatState:
    response = runner.run(state["messages"][-1])
    return {"messages": state["messages"] + [response]}


reporter = Reporter()


def call_reporter(state: ChatState) -> ChatState:
    response = reporter.report(state["messages"][-1])
    return {"messages": state["messages"] + [response]}


@flow(log_prints=True, name="Unit of Work")
def do_work(prompt: str):
    print("STARTING WORK")

    print("Compiling state graph...")
    graph = StateGraph(ChatState)
    graph.add_node("planner", call_planner)
    graph.add_edge(START, "planner")
    graph.add_node("runner", call_runner)
    app = graph.compile()

    print("Graph compiled, invoking planner...")
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})

    print("Planner invoked, processing result...")
    response = result["messages"]
    fp = Path("/home/ctbus/Penn/VibeBfx/out.txt")
    with fp.open("w", encoding="utf-8") as fh:
        for r in response:
            fh.write(r.content + "\n")

    return response

import logging
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
    msg = HumanMessage(content=f"script: {response.script}\nenv: {response.env}")
    return {"messages": state["messages"] + [msg]}


reporter = Reporter()


def call_reporter(state: ChatState) -> ChatState:
    response = reporter.report(state["messages"][-1])
    msg = HumanMessage(content=response.summary)
    return {"messages": state["messages"] + [msg]}


@flow(log_prints=True, flow_run_name="{project}_{task}")
def do_work(prompt: str, project: str, task: str):
    logging.info("STARTING WORK")

    logging.info("Compiling state graph...")
    graph = StateGraph(ChatState)
    graph.add_node("planner", call_planner)
    graph.add_edge(START, "planner")
    graph.add_node("runner", call_runner)
    graph.add_edge("planner", "runner")
    graph.add_node("reporter", call_reporter)
    graph.add_edge("runner", "reporter")
    graph.add_edge("reporter", END)
    app = graph.compile()

    print("Graph compiled, invoking planner...")
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})

    print("Planner invoked, processing result...")
    messages = result["messages"]
    task_dir = Path(project) / task
    task_dir.mkdir(parents=True, exist_ok=True)
    fp = task_dir / "out.txt"
    with fp.open("w", encoding="utf-8") as fh:
        for r in messages:
            fh.write(r.content + "\n")

    return messages

import json
import logging
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pathlib import Path
from prefect import flow, task
from typing import List, TypedDict
from vibe_bfx.agents import Planner, Reporter, Runner


class ChatState(TypedDict):
    messages: List[BaseMessage]
    steps: List[str]
    current_step: int


planner = Planner()


def call_planner(state: ChatState) -> ChatState:
    response = planner.make_plan(state["messages"][-1])
    steps = response.steps
    plan_message = AIMessage(content="\n".join(steps))
    next_msg = HumanMessage(content=steps[0]) if steps else HumanMessage(content="")
    return {
        "messages": state["messages"] + [plan_message, next_msg],
        "steps": steps,
        "current_step": 0,
    }


runner = Runner()


def call_runner(state: ChatState) -> ChatState:
    response = runner.run(state["messages"][-1])
    msg = HumanMessage(content=json.dumps({"script": response.script, "env": response.env}))
    return {
        **state,
        "messages": state["messages"] + [msg],
    }


reporter = Reporter()


def call_reporter(state: ChatState) -> ChatState:
    response = reporter.report(state["messages"][-1])
    msg = HumanMessage(content=response.summary)
    next_index = state["current_step"] + 1
    new_messages = state["messages"] + [msg]
    if next_index < len(state["steps"]):
        new_messages.append(HumanMessage(content=state["steps"][next_index]))
    return {
        "messages": new_messages,
        "steps": state["steps"],
        "current_step": next_index,
    }


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

    def continue_or_end(state: ChatState):
        return "runner" if state["current_step"] < len(state["steps"]) else END

    graph.add_conditional_edges("reporter", continue_or_end)
    app = graph.compile()

    print("Graph compiled, invoking planner...")
    result = app.invoke({"messages": [HumanMessage(content=prompt)], "steps": [], "current_step": 0})

    print("Planner invoked, processing result...")
    messages = result["messages"]
    task_dir = Path(project) / task
    task_dir.mkdir(parents=True, exist_ok=True)
    fp = task_dir / "out.txt"
    with fp.open("w", encoding="utf-8") as fh:
        for r in messages:
            fh.write(r.content + "\n")

    return messages

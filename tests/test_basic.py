from __future__ import annotations

from pathlib import Path

from vibe_bfx import Project
from langchain.schema import AIMessage


class EchoModel:
    def invoke(self, messages):
        return AIMessage(content=messages[-1].content)


def test_project_and_task(tmp_path: Path):
    project = Project(tmp_path / "proj")
    task = project.create_task("t1")
    assert (task.chat_file).exists()
    assert (task.log_file).exists()

    task.append_chat("user", "hi")
    task.append_log("started")

    assert "user: hi" in task.chat_file.read_text()
    assert "started" in task.log_file.read_text()


def test_run_chat(tmp_path: Path):
    project = Project(tmp_path / "proj")
    task = project.create_task("t1")
    response = task.run_chat("echo", model=EchoModel())
    assert response == "echo"
    assert "assistant: echo" in task.chat_file.read_text()

    # ensure node logs are created and referenced
    node_logs = list((task.path / "logs").glob("*.log"))
    assert node_logs, "no node log files created"
    log_refs = task.log_file.read_text().splitlines()
    assert any("logs/" in line for line in log_refs)

from __future__ import annotations

from pathlib import Path

from vibe_bfx import Planner, Project


def add(a: int, b: int) -> int:
    return a + b


def test_planner_uses_workers(tmp_path: Path) -> None:
    project = Project(tmp_path / "proj")
    task = project.create_task("t1")
    planner = Planner(task)

    report = planner.run(tool=add, inputs={"a": 1, "b": 2})

    assert "Result: 3" in report

    log_text = task.log_file.read_text()
    assert "environment: docker" in log_text
    assert "execution result: 3" in log_text
    assert "Result: 3" in log_text

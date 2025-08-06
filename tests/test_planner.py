from __future__ import annotations

from pathlib import Path

from vibe_bfx import Project


def add(a: int, b: int) -> int:
    return a + b


def test_planner_runs_steps(tmp_path: Path) -> None:
    project = Project(tmp_path / "proj")
    task = project.create_task("t1")
    report = task.run(
        "add numbers",
        tool=add,
        inputs={"a": 1, "b": 2},
    )

    assert "add numbers: 3" in report

    chat_text = task.chat_file.read_text()
    assert "user: add numbers" in chat_text
    assert "assistant: add numbers: 3" in chat_text

    # log references
    refs = task.log_file.read_text().splitlines()
    assert any("planner" in r for r in refs)
    assert any("runner" in r for r in refs)
    assert any("analyzer" in r for r in refs)

    # verify log contents
    planner_log = next(task.logs_dir.glob("*_planner.log")).read_text()
    assert "plan" in planner_log
    runner_log = next(task.logs_dir.glob("*_runner.log")).read_text()
    assert "result: 3" in runner_log
    analyzer_log = next(task.logs_dir.glob("*_analyzer.log")).read_text()
    assert "add numbers: 3" in analyzer_log

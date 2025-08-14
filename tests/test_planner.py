from __future__ import annotations

from pathlib import Path

from vibe_bfx import Project


def test_planner_uses_workers(tmp_path: Path) -> None:
    project = Project(tmp_path / "proj")
    task = project.create_task("t1")
    cmd = [
        "python",
        "-c",
        "import sys; print(int(sys.argv[1]) + int(sys.argv[2]))",
    ]
    report = task.run(
        "add numbers",
        command=cmd,
        inputs={"a": 1, "b": 2},
    )

    assert "Result: 3" in report

    chat_text = task.chat_file.read_text()
    assert "user: add numbers" in chat_text
    assert "assistant: Result: 3" in chat_text

    # log references
    refs = task.log_file.read_text().splitlines()
    assert any("planner" in r for r in refs)
    assert any("environment" in r for r in refs)
    assert any("executor" in r for r in refs)
    assert any("analyst" in r for r in refs)

    # verify log contents
    env_log = next((task.logs_dir.glob("*_environment.log"))).read_text()
    assert "environment: docker" in env_log
    exec_log = next((task.logs_dir.glob("*_executor.log"))).read_text()
    assert "execution result: 3" in exec_log
    analyst_log = next((task.logs_dir.glob("*_analyst.log"))).read_text()
    assert "Result: 3" in analyst_log

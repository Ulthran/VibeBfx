from langchain.schema import HumanMessage
from vibe_bfx.agents import RunResponse, ReportResponse


def test_pipeline_with_mocks(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from vibe_bfx import prefect as wf

    def fake_plan(prompt):
        return HumanMessage(content="[step1]")

    def fake_run(prompt):
        return RunResponse(script="echo 1", env="bash")

    def fake_report(prompt):
        return ReportResponse(summary="done")

    monkeypatch.setattr(wf.planner, "make_plan", fake_plan)
    monkeypatch.setattr(wf.runner, "run", fake_run)
    monkeypatch.setattr(wf.reporter, "report", fake_report)

    messages = wf.do_work("hello", str(tmp_path), "task")
    contents = [m.content for m in messages]
    assert contents == ["hello", "[step1]", "script: echo 1\nenv: bash", "done"]

    out_fp = tmp_path / "task" / "out.txt"
    assert out_fp.exists()
    assert out_fp.read_text().strip().splitlines()[-1] == "done"

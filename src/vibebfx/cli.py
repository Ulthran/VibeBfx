"""Typer-based CLI for VibeBfx DeepAgents workflows."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich import print

from .agents import DeepAgentOrchestrator
from .config import AppConfig
from .tasks import BioTask

app = typer.Typer(add_completion=False)


def _load_llm(model: str):
    """Load a chat model from langchain-openai if available."""

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise typer.BadParameter(
            "langchain-openai is required to run agents. Install with `pip install langchain-openai`."
        ) from exc

    return ChatOpenAI(model=model, temperature=0)


@app.command()
def config() -> None:
    """Show the resolved configuration (including env overrides)."""

    cfg = AppConfig()
    print(cfg.model_dump_json(indent=2))


@app.command()
def run(
    description: str = typer.Argument(..., help="Human-readable description of the bio task."),
    dataset: str = typer.Option(..., "--dataset", help="Dataset path or dataset name inside the dataset root."),
    params: str = typer.Option("{}", "--params", help="JSON string of task parameters."),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Optional job id override."),
    model: Optional[str] = typer.Option(None, "--model", help="Chat model name; defaults to config.default_model."),
    execute: bool = typer.Option(
        False,
        "--execute/--plan-only",
        help="Run the DeepAgent immediately (requires model + API key).",
    ),
) -> None:
    """Submit a job and optionally run the DeepAgent."""

    cfg = AppConfig()
    orchestrator = DeepAgentOrchestrator(cfg)

    try:
        parameters = json.loads(params)
    except json.JSONDecodeError as exc:  # pragma: no cover - validated by Typer at runtime
        raise typer.BadParameter(f"Invalid JSON for params: {exc}") from exc

    task = BioTask(description=description, dataset=dataset, parameters=parameters)

    if not execute:
        record = orchestrator.workspace_manager.create_job(task, job_id=job_id)
        print(f"Prepared workspace at {record.workspace} (job {record.job_id}). Use --execute to run the agent.")
        return

    llm = _load_llm(model or cfg.default_model)
    record = orchestrator.run_job(llm, task, job_id=job_id)
    print(f"Job {record.job_id} finished with status {record.status}. Workspace: {record.workspace}")
    if record.summary:
        print(f"Summary: {record.summary}")


if __name__ == "__main__":  # pragma: no cover
    app()

"""DeepAgents orchestration helpers built on LangChain."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from rich.console import Console

from .config import AppConfig
from .tasks import BioTask, JobRecord
from .workspace import WorkspaceManager

SYSTEM_PROMPT = """
You are a DeepAgent orchestrating bioinformatics workflows. Each job has an isolated workspace on disk.
- Use the provided shell and python tools to run real commands against the workspace.
- Never assume datasets exist: verify paths under the dataset mount or download them.
- Plan the pipeline (download -> QC -> processing -> analysis -> report) before execution and keep a concise todo list.
- Persist outputs inside the workspace (artifacts/) and log progress to logs/job.log.
- Return a short summary plus the path to the workspace when done.
"""


@dataclass
class SubAgentSpec:
    """Configuration for a sub-agent that handles a pipeline phase."""

    name: str
    objective: str
    tools: Sequence[BaseTool] = field(default_factory=list)


class DeepAgentOrchestrator:
    """Builds DeepAgent executors with shared workspace-aware tools."""

    def __init__(self, config: Optional[AppConfig] = None, console: Optional[Console] = None):
        self.config = config or AppConfig()
        self.console = console or Console()
        self.workspace_manager = WorkspaceManager(self.config)

    def _log_to_file(self, record: JobRecord, message: str) -> None:
        log_path = self.workspace_manager.job_log_path(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def _default_tools(self, record: JobRecord, dataset_mount: Path) -> List[BaseTool]:
        workspace = record.workspace
        log = self._log_to_file

        @tool("workspace_shell", return_direct=False)
        def workspace_shell(command: str) -> str:
            """Execute a shell command inside the job workspace."""

            process = subprocess.run(
                command,
                cwd=workspace,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
            )
            stdout = process.stdout.strip()
            stderr = process.stderr.strip()
            if stdout:
                log(record, f"[shell stdout]\n{stdout}")
            if stderr:
                log(record, f"[shell stderr]\n{stderr}")
            return json.dumps(
                {
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )

        @tool("python_runner", return_direct=False)
        def python_runner(code: str) -> str:
            """Run a Python snippet inside the job workspace."""

            local_env: dict[str, Any] = {
                "workspace": workspace,
                "dataset_mount": dataset_mount,
            }
            try:
                exec(code, local_env)
                result = local_env.get("result")
            except Exception as exc:  # pragma: no cover - handled at runtime
                result = f"Python execution failed: {exc}"
            log(record, f"[python]\n{code}\n[result]\n{result}")
            return json.dumps({"result": str(result)})

        @tool("workspace_note", return_direct=True)
        def workspace_note(note: str) -> str:
            """Write a note to the job log for traceability."""

            log(record, f"[note]\n{note}")
            return note

        return [workspace_shell, python_runner, workspace_note]

    def _build_prompt(self, task: BioTask, record: JobRecord, dataset_mount: Path) -> ChatPromptTemplate:
        todo_hint = "Planning is enabled; create a todo list and update it as you run the pipeline." if self.config.planning_enabled else "Planning is disabled; act directly but still keep logs concise."
        return ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT.strip()),
                ("system", f"Workspace: {record.workspace}\nDataset mount: {dataset_mount}"),
                ("system", todo_hint),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    def build_executor(
        self,
        llm: BaseChatModel,
        task: BioTask,
        record: JobRecord,
        dataset_mount: Path,
        subagents: Optional[Sequence[SubAgentSpec]] = None,
    ) -> AgentExecutor:
        tools: List[BaseTool] = self._default_tools(record, dataset_mount)
        for spec in subagents or []:
            tools.extend(spec.tools)

        prompt = self._build_prompt(task, record, dataset_mount)
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def run_job(
        self,
        llm: BaseChatModel,
        task: BioTask,
        job_id: Optional[str] = None,
        subagents: Optional[Sequence[SubAgentSpec]] = None,
    ) -> JobRecord:
        record = self.workspace_manager.create_job(task, job_id=job_id)
        dataset_mount = self.workspace_manager.dataset_mount(self.config, task)
        executor = self.build_executor(llm, task, record, dataset_mount, subagents=subagents)

        user_prompt = (
            "You are handling a bioinformatics request. Build a todo list, execute steps, and summarize results.\n"
            f"Dataset: {task.dataset}\n"
            f"Parameters: {json.dumps(task.parameters)}"
        )

        self.console.print(f"Running DeepAgent for job {record.job_id} in {record.workspace}")
        record.mark_running()
        record.persist()
        try:
            output = executor.invoke({"messages": [user_prompt]})
            summary = output.get("output", "Job completed.") if isinstance(output, dict) else str(output)
            result_dir = self.workspace_manager.results_dir(record)
            record.mark_complete(result_dir, summary)
        except Exception as exc:  # pragma: no cover - runtime failure path
            summary = f"Job failed: {exc}"
            record.mark_failed(summary)
        finally:
            record.persist()
        return record

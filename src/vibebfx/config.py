"""Configuration for the VibeBfx library."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Runtime configuration for DeepAgents bioinformatics workflows."""

    workspace_root: Path = Field(
        default_factory=lambda: Path.cwd() / "jobs",
        description="Root directory where per-job workspaces are created.",
    )
    dataset_root: Path = Field(
        default_factory=lambda: Path.cwd() / "datasets",
        description="Shared dataset directory that agents can mount read-only into jobs.",
    )
    default_model: str = Field(
        default="gpt-4o-mini",
        description="Identifier for the default chat model used by LangChain agents.",
    )
    planning_enabled: bool = Field(
        default=True,
        description="Enable agent todo-list planning before execution.",
    )
    max_subagents: int = Field(
        default=4,
        description="Maximum number of sub-agents spawned for a job.",
    )
    log_stdout: bool = Field(
        default=True,
        description="Stream shell tool stdout/stderr into the job log file for observability.",
    )

    class Config:
        env_prefix = "VIBEBFX_"
        arbitrary_types_allowed = True

    def ensure_directories(self) -> None:
        """Create required directories if they do not exist."""

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.dataset_root.mkdir(parents=True, exist_ok=True)

    def workspace_for(self, job_id: str) -> Path:
        """Return the workspace path for a given job id."""

        return self.workspace_root / job_id


DEFAULT_CONFIG = AppConfig()

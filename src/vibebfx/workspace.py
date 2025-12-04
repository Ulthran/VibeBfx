"""Workspace management for isolated bioinformatics jobs."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .tasks import BioTask, JobRecord


class WorkspaceManager:
    """Provision and manage per-job workspaces."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.config.ensure_directories()

    def create_job(self, task: BioTask, job_id: Optional[str] = None) -> JobRecord:
        job_id = job_id or uuid.uuid4().hex
        workspace = self.config.workspace_for(job_id)
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "logs").mkdir(exist_ok=True)
        (workspace / "artifacts").mkdir(exist_ok=True)

        record = JobRecord(job_id=job_id, task=task, workspace=workspace)
        record.persist()
        return record

    @staticmethod
    def job_log_path(record: JobRecord) -> Path:
        return record.workspace / "logs" / "job.log"

    @staticmethod
    def results_dir(record: JobRecord) -> Path:
        results = record.workspace / "artifacts"
        results.mkdir(exist_ok=True)
        return results

    @staticmethod
    def dataset_mount(config: AppConfig, task: BioTask) -> Path:
        candidate = Path(task.dataset)
        return candidate if candidate.is_absolute() else config.dataset_root / candidate

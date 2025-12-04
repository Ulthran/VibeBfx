"""Task and job metadata models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(slots=True)
class BioTask:
    """Bioinformatics job request from a user."""

    description: str
    dataset: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None


@dataclass(slots=True)
class JobRecord:
    """Tracks runtime metadata for a job workspace."""

    job_id: str
    task: BioTask
    workspace: Path
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"
    result_path: Optional[Path] = None
    summary: Optional[str] = None

    def mark_running(self) -> None:
        self.status = "running"

    def mark_complete(self, result_path: Path, summary: str) -> None:
        self.status = "completed"
        self.result_path = result_path
        self.summary = summary

    def mark_failed(self, summary: str) -> None:
        self.status = "failed"
        self.summary = summary

    def to_json(self) -> str:
        payload = {
            "job_id": self.job_id,
            "task": {
                "description": self.task.description,
                "dataset": self.task.dataset,
                "parameters": self.task.parameters,
                "user_id": self.task.user_id,
            },
            "workspace": str(self.workspace),
            "created_at": self.created_at.isoformat() + "Z",
            "status": self.status,
            "result_path": str(self.result_path) if self.result_path else None,
            "summary": self.summary,
        }
        return json.dumps(payload, indent=2)

    def persist(self) -> Path:
        log_path = self.workspace / "job.json"
        log_path.write_text(self.to_json())
        return log_path

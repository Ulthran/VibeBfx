from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import csv
from typing import List

import yaml

from .task import Task


class Project:
    """Represents a top-level project directory.

    Parameters
    ----------
    path: str | Path
        Location of the project directory. If it does not exist it will be
        created automatically.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.path / "metadata.csv"
        self.config_path = self.path / "config.yaml"
        self.output_dir = self.path / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = self._load_metadata()
        self.config = self._load_config()

    def _load_metadata(self) -> List[dict[str, str]]:
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                return list(reader)
        return []

    def _load_config(self) -> dict[str, Any]:
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            return data or {}
        return {}

    def create_task(self, name: str) -> Task:
        """Create and return a task within this project."""
        task_path = self.path / name
        task_path.mkdir(parents=True, exist_ok=True)
        return Task(task_path)

    def get_task(self, name: str) -> Optional[Task]:
        """Return a task if it exists, otherwise ``None``."""
        task_path = self.path / name
        if task_path.exists() and task_path.is_dir():
            return Task(task_path)
        return None

    def list_tasks(self) -> Iterable[str]:
        """Yield the names of tasks in this project."""
        for p in sorted(self.path.iterdir()):
            if p.is_dir() and p != self.output_dir and not p.name.startswith("."):
                yield p.name

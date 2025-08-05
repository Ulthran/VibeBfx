from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

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
            if p.is_dir():
                yield p.name

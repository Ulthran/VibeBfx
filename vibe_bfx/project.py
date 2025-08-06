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

    # Reserved configuration keys and their default values. These are
    # automatically populated in ``project.config`` to provide a stable API
    # regardless of whether users explicitly set them in ``config.yaml``.
    _RESERVED_CONFIG_DEFAULTS: dict[str, Any] = {
        "db_fp": None,
        "conda_fp": None,
        "docker_fp": None,
        "singularity_fp": None,
        # The column names in ``metadata.csv`` that correspond to the sample ID
        # and the R1 fastq path. Users may override these in ``config.yaml`` if
        # their metadata file uses non-standard column names.
        "sample_id_field_name": "sample_id",
        "r1_fp_field_name": "r1_fp",
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.path / "metadata.csv"
        self.config_path = self.path / "config.yaml"
        self.output_dir = self.path / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = self._load_metadata()
        self.config = self._load_config()
        # Ensure all reserved keys are present in the configuration with their
        # default values.
        for key, default in self._RESERVED_CONFIG_DEFAULTS.items():
            self.config.setdefault(key, default)

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

    def iter_samples(self) -> Iterable[dict[str, str]]:
        """Yield dictionaries containing ``sample_id`` and ``r1_fp`` for each
        record in the metadata.

        The ``sample_id`` and ``r1_fp`` column names can be overridden via the
        ``sample_id_field_name`` and ``r1_fp_field_name`` configuration keys.
        """

        sample_key = self.config["sample_id_field_name"]
        r1_key = self.config["r1_fp_field_name"]
        for row in self.metadata:
            if sample_key in row and r1_key in row:
                yield {"sample_id": row[sample_key], "r1_fp": row[r1_key]}

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
            if p.is_dir() and p != self.output_dir and not p.name.startswith('.'):
                yield p.name

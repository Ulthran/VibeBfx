from pathlib import Path
import json

from vibebfx.config import AppConfig
from vibebfx.tasks import BioTask
from vibebfx.workspace import WorkspaceManager


def test_create_job_creates_directories_and_log(tmp_path: Path) -> None:
    config = AppConfig(workspace_root=tmp_path / "jobs", dataset_root=tmp_path / "datasets")
    manager = WorkspaceManager(config)
    task = BioTask(description="RNA-seq QC", dataset="rna", parameters={"adapter": "AGAT"})

    record = manager.create_job(task, job_id="job123")

    assert record.workspace.exists()
    assert (record.workspace / "logs").exists()
    assert (record.workspace / "artifacts").exists()
    metadata = json.loads((record.workspace / "job.json").read_text())
    assert metadata["job_id"] == "job123"
    assert metadata["task"]["description"] == "RNA-seq QC"


def test_dataset_mount_resolves_relative_paths(tmp_path: Path) -> None:
    config = AppConfig(workspace_root=tmp_path / "jobs", dataset_root=tmp_path / "datasets")
    manager = WorkspaceManager(config)
    task = BioTask(description="Demo", dataset="rna", parameters={})

    mount = manager.dataset_mount(config, task)
    assert mount == config.dataset_root / "rna"

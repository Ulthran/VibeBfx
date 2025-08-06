from __future__ import annotations

from pathlib import Path
import logging

from vibe_bfx import Project
from langchain.schema import AIMessage


class EchoModel:
    def invoke(self, messages):
        return AIMessage(content=messages[-1].content)


def test_project_and_task(tmp_path: Path):
    proj_root = tmp_path / "proj"
    proj_root.mkdir()
    (proj_root / "metadata.csv").write_text("sample,file\nA,foo.txt\n")
    (proj_root / "config.yaml").write_text("db: /data/db\n")

    project = Project(proj_root)
    assert project.metadata == [{"sample": "A", "file": "foo.txt"}]
    assert project.config["db"] == "/data/db"
    assert project.output_dir.exists()

    task = project.create_task("t1")
    assert task.path == project.path / "t1"
    assert task.chat_file.exists()
    assert task.log_file.exists()

    task.append_chat("user", "hi")
    task.append_log("started")

    assert "user: hi" in task.chat_file.read_text()
    assert "started" in task.log_file.read_text()


def test_reserved_config_defaults(tmp_path: Path):
    proj_root = tmp_path / "proj"
    proj_root.mkdir()
    # metadata uses default column names
    (proj_root / "metadata.csv").write_text(
        "sample_id,r1_fp\nS1,reads.fastq\n"
    )

    project = Project(proj_root)

    # All reserved keys should be present with default values
    assert project.config["db_fp"] is None
    assert project.config["conda_fp"] is None
    assert project.config["docker_fp"] is None
    assert project.config["singularity_fp"] is None
    assert project.config["sample_id_field_name"] == "sample_id"
    assert project.config["r1_fp_field_name"] == "r1_fp"

    # iter_samples should expose standardized keys
    samples = list(project.iter_samples())
    assert samples == [{"sample_id": "S1", "r1_fp": "reads.fastq"}]


def test_custom_metadata_fields(tmp_path: Path):
    proj_root = tmp_path / "proj"
    proj_root.mkdir()
    # metadata uses custom column names
    (proj_root / "metadata.csv").write_text("sid,read1\nA,foo.fq\n")
    # override the column names via config
    (proj_root / "config.yaml").write_text(
        "sample_id_field_name: sid\nr1_fp_field_name: read1\n"
    )

    project = Project(proj_root)
    samples = list(project.iter_samples())
    assert samples == [{"sample_id": "A", "r1_fp": "foo.fq"}]


def test_run_chat(tmp_path: Path):
    project = Project(tmp_path / "proj")
    task = project.create_task("t1")
    response = task.run_chat("echo", model=EchoModel())
    assert response == "echo"
    assert "assistant: echo" in task.chat_file.read_text()

    # ensure node logs are created and referenced
    node_logs = list((task.path / "logs").glob("*.log"))
    assert node_logs, "no node log files created"
    log_refs = task.log_file.read_text().splitlines()
    assert any("logs/" in line for line in log_refs)


def test_iter_samples_logs_skipped(tmp_path: Path, caplog):
    proj_root = tmp_path / "proj"
    proj_root.mkdir()
    (proj_root / "metadata.csv").write_text(
        "sample_id,r1_fp\nS1,reads1.fq\nS2,\n,reads3.fq\nS3,reads4.fq\n"
    )

    project = Project(proj_root)
    with caplog.at_level(logging.WARNING):
        samples = list(project.iter_samples())

    assert samples == [
        {"sample_id": "S1", "r1_fp": "reads1.fq"},
        {"sample_id": "S3", "r1_fp": "reads4.fq"},
    ]
    assert "Skipped metadata rows missing required fields: [2, 3]" in caplog.text

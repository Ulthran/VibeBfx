from __future__ import annotations

from celery import Celery
import subprocess
from typing import Sequence

# The Celery application used throughout the project.  It defaults to an in
# memory broker/back end and eager execution so that unit tests do not require
# a separate worker process.  In production these settings can be overridden
# by environment variables or a configuration file.
app = Celery("vibe_bfx", broker="memory://", backend="rpc://")
app.conf.update(
    task_always_eager=True,
    task_store_eager_result=True,
    task_serializer="pickle",
    result_serializer="pickle",
    accept_content=["pickle"],
)


@app.task(name="vibe_bfx.execute_tool")
def execute_tool(command: Sequence[str]) -> str:
    """Run ``command`` via :mod:`subprocess` and return its stdout."""

    completed = subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()

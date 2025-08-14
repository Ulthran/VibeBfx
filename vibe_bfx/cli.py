from __future__ import annotations

import argparse
import shlex

from .project import Project
from .celery_app import execute_tool


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command via Celery")
    parser.add_argument("project", help="Project directory")
    parser.add_argument("task", help="Task name inside the project")
    parser.add_argument("--prompt", required=True, help="Command to execute")
    args = parser.parse_args()

    project = Project(args.project)
    project.get_task(args.task) or project.create_task(args.task)

    command = shlex.split(args.prompt)
    result = execute_tool.delay(command).get()
    print(result)


if __name__ == "__main__":  # pragma: no cover
    main()

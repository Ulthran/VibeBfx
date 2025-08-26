import argparse
import logging
from pathlib import Path
from vibe_bfx.prefect import do_work


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command")
    parser.add_argument("project", help="Project directory")
    parser.add_argument("task", help="Task name inside the project")
    parser.add_argument("--prompt", required=True, help="Command to execute")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    logging.info(
        f"Running command in project: {args.project}, task: {args.task}, prompt: {args.prompt}"
    )
    do_work(args.prompt, Path(args.project).name, args.task)
    return


if __name__ == "__main__":  # pragma: no cover
    main()

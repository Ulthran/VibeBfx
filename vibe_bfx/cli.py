import argparse
from vibe_bfx.celery_app import do_work


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command via Celery")
    parser.add_argument("project", help="Project directory")
    parser.add_argument("task", help="Task name inside the project")
    parser.add_argument("--prompt", required=True, help="Command to execute")
    args = parser.parse_args()

    print(
        f"Running command in project: {args.project}, task: {args.task}, prompt: {args.prompt}"
    )
    do_work.apply_async((args.prompt,))
    return


if __name__ == "__main__":  # pragma: no cover
    main()

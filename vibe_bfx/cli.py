from __future__ import annotations

import argparse

from .project import Project


def main() -> None:
    parser = argparse.ArgumentParser(description="Vibe BFX text interface")
    parser.add_argument("project", help="Project directory")
    parser.add_argument("task", help="Task name inside the project")
    parser.add_argument("--prompt", help="Optional prompt for a single turn")
    args = parser.parse_args()

    project = Project(args.project)
    task = project.get_task(args.task) or project.create_task(args.task)

    if args.prompt:
        response = task.run_chat(args.prompt)
        print(response)
        return

    try:
        while True:
            prompt = input("> ")
            if not prompt.strip():
                continue
            response = task.run_chat(prompt)
            print(response)
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":  # pragma: no cover
    main()

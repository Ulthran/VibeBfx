from __future__ import annotations

from .celery_app import app


def main() -> None:
    """Start a Celery worker using the project's application."""
    app.worker_main(["worker", "--loglevel=info"])


if __name__ == "__main__":  # pragma: no cover
    main()

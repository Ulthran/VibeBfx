# VibeBfx

VibeBfx is a fresh Python library for building DeepAgents-powered bioinformatics workflows with LangChain. Each submitted task runs in an isolated workspace on disk, enabling reproducible pipelines that combine shell tools, Python analytics, and modular sub-agents.

## Key concepts
- **Isolated workspaces:** every job receives a dedicated root directory for downloads, intermediate artifacts, and outputs that persist after the run.
- **DeepAgents orchestration:** a job agent composes shell execution, Python analysis helpers, and optional sub-agents (download/QC/analysis/report) built with LangChain's tool-calling agents.
- **Planning-first flow:** the agent decomposes tasks into a todo list, executes steps, logs progress, and returns a structured result summary alongside the workspace path.
- **Config-driven:** shared dataset roots, default models, and resource settings are configurable through a Pydantic model with environment variable overrides.

VibeBfx targets LangChain `1.1.x` and DeepAgents `0.2.x` to align with the latest tool-calling and sub-agent capabilities.

## Quick start
1. Install dependencies (requires Python 3.10+):
   ```bash
   pip install -e .
   ```
2. Submit a job via the CLI:
   ```bash
   vibe-bfx run "Run RNA-seq QC + differential expression" --dataset /data/rnaseq/exp1 \
     --params '{"organism": "human", "adapter": "AGATCGGA"}'
   ```
3. Outputs land in a unique workspace under the configured `workspace_root` (default: `./jobs`). Logs and summaries are written to `job.json` inside that directory.

## Library building blocks
- `vibebfx.config.AppConfig` loads workspace, dataset, and model settings.
- `vibebfx.workspace.WorkspaceManager` provisions per-job directories and standard paths for logs and results.
- `vibebfx.tasks.BioTask` captures job metadata, while `JobRecord` tracks runtime status and outputs.
- `vibebfx.agents.DeepAgentOrchestrator` builds LangChain tool-calling agents with shell and Python execution tools plus optional sub-agent specs.
- `vibebfx.cli` exposes a Typer-based CLI for running jobs, inspecting configuration, and preparing workspaces.

## Notes
This repository intentionally focuses on orchestrating real bioinformatics pipelines. Tools run against the persistent filesystem, not an ephemeral agent-state virtual FS, to accommodate large outputs and reproducible reruns.

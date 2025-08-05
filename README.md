# VibeBfx

A minimal framework for agentic "vibe bioinformatics".  Projects are
simple directories and each unit of work is a *task* stored in a
subdirectory containing a `chat.txt` conversation history and a `log.txt`
file with notes or other metadata.

The core library is built on top of [LangChain](https://python.langchain.com)
and [LangGraph](https://langchain-ai.github.io/langgraph/) to provide a
text interface reminiscent of Codex.

## Quick start

```bash
pip install -e .

# Start an interactive session for a project and task
vibe-bfx my_project first-task --prompt "hello"
```

Running the command above will create directories
`my_project/first-task/chat.txt` and `my_project/first-task/log.txt` which
store the conversation and log respectively.

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

# Start redis backend
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# Start VibeBfx worker
vibe-bfx-worker

# In another terminal, start a task
vibe-bfx my_project first-task --prompt "hello"
```

Running the command above will create directories
`my_project/first-task/chat.txt` and `my_project/first-task/log.txt` which
store the conversation and log respectively.

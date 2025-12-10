# VibeBfx

A minimal framework for agentic "vibe bioinformatics".  Projects are
simple directories and each unit of work is a *task* stored in a
subdirectory containing a `chat.txt` conversation history and a `log.txt`
file with notes or other metadata.

Each task will be carried out by a LangChain DeepAgent and will make use 
of tools provided by an MCP server providing a machine-friendly interface 
to all the data on the filesystem.
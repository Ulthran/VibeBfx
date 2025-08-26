from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TypedDict

import logging
from datetime import datetime
from contextlib import contextmanager

from langchain.schema import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI


class ChatState(TypedDict):
    messages: List[BaseMessage]


class Task:
    """Represents a unit of work inside a project."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

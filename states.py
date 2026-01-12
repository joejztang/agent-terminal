from typing import TypedDict

from langgraph.graph import MessagesState


class SimpleState(MessagesState):
    decision: str

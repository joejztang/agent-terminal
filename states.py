from typing import TypedDict

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    decision: str
    verbose: bool

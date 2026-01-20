import operator
from typing import Annotated, List, TypedDict

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    decision: str
    verbose: bool
    intent: Annotated[list[str], operator.add]
    urls: Annotated[list[str], operator.add]

from typing import Any

from langgraph.graph import END, START, StateGraph

from flow_controls import simple_after_ai_response, simple_after_ask
from nodes import ai_response, ask, initialize_state
from states import SimpleState
from util import EXIT


def simple_chat(llm: Any) -> StateGraph:
    graph = StateGraph(SimpleState)

    # graph.add_node("initialize_state", initialize_state)
    graph.add_node("ask", ask)
    graph.add_node("ai_response", lambda state: ai_response(state, llm))

    # graph.add_edge(START, "initialize_state")
    # graph.add_edge("initialize_state", "ask")
    graph.add_edge(START, "ask")
    graph.add_conditional_edges(
        "ask",
        simple_after_ask,
        {
            "ai_response": "ai_response",
            EXIT: END,
        },
    )

    graph.add_conditional_edges(
        "ai_response",
        simple_after_ai_response,
        {
            "ask": "ask",
            EXIT: END,
        },
    )
    return graph

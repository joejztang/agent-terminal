from typing import Any

from langgraph.graph import END, START, StateGraph

from flow_controls import after_router, simple_after_ask, simple_after_control
from nodes import (
    ai_response,
    ask,
    fetch_from_vectordb,
    initialize_state,
    router,
    upload_to_vectordb,
)
from states import SimpleState
from util import CONTINUE, EXIT


def simple_chat(llm: Any) -> StateGraph:
    graph = StateGraph(SimpleState)

    # graph.add_node("initialize_state", initialize_state)
    graph.add_node("router", lambda state: router(state, llm))
    graph.add_node("ask", ask)
    graph.add_node("ai_response", lambda state: ai_response(state, llm))
    graph.add_node("upload_to_vectordb", upload_to_vectordb)
    graph.add_node("fetch_from_vectordb", fetch_from_vectordb)

    # graph.add_edge(START, "initialize_state")
    # graph.add_edge("initialize_state", "ask")
    graph.add_edge(START, "ask")
    graph.add_edge("ask", "router")
    graph.add_conditional_edges(
        "router",
        after_router,
        {
            EXIT: END,
            CONTINUE: "ask",
            "upload_to_vectordb": "upload_to_vectordb",
            "fetch_from_vectordb": "fetch_from_vectordb",
            "none": "ai_response",
        },
    )
    # graph.add_conditional_edges(
    #     "ask",
    #     simple_after_ask,
    #     {
    #         "ai_response": "ai_response",
    #         CONTINUE: "ask",
    #         EXIT: END,
    #     },
    # )

    graph.add_conditional_edges(
        "ai_response",
        simple_after_control,
        {
            "ask": "ask",
            CONTINUE: "ask",
            EXIT: END,
        },
    )
    graph.add_conditional_edges(
        "upload_to_vectordb",
        simple_after_control,
        {
            "ask": "ask",
            CONTINUE: "ask",
            EXIT: END,
        },
    )
    graph.add_conditional_edges(
        "fetch_from_vectordb",
        simple_after_control,
        {
            "ask": "ask",
            CONTINUE: "ask",
            EXIT: END,
        },
    )
    return graph

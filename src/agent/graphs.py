from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.flow_controls import after_router, simple_after_ask, simple_after_control
from src.agent.states import AgentState
from src.nodes.nodes import (  # tool_decision,
    ai_response,
    fetch_from_vectordb,
    router,
    tool_response,
    upload_html_to_vectordb,
)
from src.utils.util import CONTINUE, EXIT


def agent_graph(llm: Any) -> StateGraph:
    graph = StateGraph(AgentState)

    # graph.add_node("initialize_state", initialize_state)
    graph.add_node("router", lambda state: router(state, llm))
    # graph.add_node("tool_decision", lambda state: tool_decision(state, llm))
    graph.add_node("ai_response", lambda state: ai_response(state, llm))
    graph.add_node("upload_to_vectordb", upload_html_to_vectordb)
    graph.add_node("fetch_from_vectordb", fetch_from_vectordb)
    graph.add_node("tool_response", tool_response)

    # graph.add_edge(START, "tool_decision")
    # graph.add_edge("tool_decision", "router")
    graph.add_edge(START, "router")
    graph.add_edge("fetch_from_vectordb", "ai_response")
    graph.add_conditional_edges(
        "router",
        after_router,
        {
            EXIT: END,
            # "show_fulltexts": "show_chroma_embedding_fulltext_search_content",
            "tool_calls": "tool_response",
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

    return graph.compile()

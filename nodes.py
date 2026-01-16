from typing import Any

from langchain.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from states import AgentState
from util import CONTINUE, EXIT, catch_interruption, config, console


@catch_interruption
def router(state: AgentState, llm: Any) -> AgentState:
    if state.get("decision") in [EXIT, CONTINUE]:
        return state

    query = state["messages"][-1].content.lower()

    system = """You are an expert on classifying customer's intention.
    If customer wants to upload a document to the vector database, respond with 'upload_to_vectordb'.
    If customer wants to fetch information from the vector database, respond with 'fetch_from_vectordb'.
    Otherwise, respond with 'none'."""
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    simple_router = route_prompt | llm
    response = simple_router.invoke({"question": query})
    return {"messages": [SystemMessage(content=response.content)]}


@catch_interruption
def upload_to_vectordb(state: AgentState) -> AgentState:
    # Placeholder for upload logic
    console.print("Uploading document to vector database...", style="yellow")
    return {"messages": []}


@catch_interruption
def fetch_from_vectordb(state: AgentState) -> AgentState:
    # Placeholder for fetch logic
    console.print("Fetching information from vector database...", style="yellow")
    return {"messages": []}


@catch_interruption
def ai_response(state: AgentState, llm: Any) -> AgentState:
    response = llm.invoke(state["messages"])

    if state.get("verbose", False):
        console.print(state, style=config.get("verbose-color"))

    return {"messages": [AIMessage(content=response.content)]}

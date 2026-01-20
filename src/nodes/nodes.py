from typing import Any, List

from langchain.messages import AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.agent.states import AgentState
from src.utils.document import process_html
from src.utils.store import vector_store
from src.utils.util import CONTINUE, EXIT, catch_interruption, config, console


class RouterOutput(BaseModel):
    query: List[str]
    intent: List[str]
    urls: List[str]


@catch_interruption
def router(state: AgentState, llm: Any) -> AgentState:
    if state.get("decision") in [EXIT, CONTINUE]:
        return state

    query = state["messages"][-1].content.lower()

    system = """You are an expert on classifying customer's intention.
    If customer wants to upload contents from an url to vector database, respond intent with 'upload_to_vectordb' and urls with the url mentioned in the content.
    If customer wants to fetch information from the vector database, respond intent with 'fetch_from_vectordb'.
    Otherwise, respond intent with 'none' and urls with 'none'."""
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    structured_llm = llm.with_structured_output(RouterOutput)
    simple_router = route_prompt | structured_llm
    response = simple_router.invoke({"question": query})
    # return {"messages": [AIMessage(content=response.content)]}
    console.print(f"Router response: {response}", style="yellow")
    return {"intent": response.intent, "urls": response.urls}


@catch_interruption
def upload_html_to_vectordb(state: AgentState) -> AgentState:
    console.print("uploading information from vector database...", style="yellow")
    return {"messages": []}
    # documents: List[Document] = process_html(state["urls"][-1])
    # vector_store.add_documents(documents)
    # return {"messages": [AIMessage(content="Document uploaded successfully.")]}


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

from typing import Any, List

from langchain.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.agent.schemas import RouterOutput
from src.agent.states import AgentState
from src.utils.document import process_html
from src.utils.store import vector_store
from src.utils.util import CONTINUE, EXIT, catch_interruption, config, console


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

    if state.get("verbose", False):
        console.print(f"Router response: {response}", style=config.get("verbose-color"))

    if response.intent[-1] == "none":
        state["content"] = []

    return {"intent": response.intent, "urls": response.urls}


@catch_interruption
def upload_html_to_vectordb(state: AgentState) -> AgentState:
    # Chroma is different from pgvector
    documents: List[Document] = process_html(state["urls"][-1])
    vector_store.add_documents(documents)
    return {"messages": [AIMessage(content="Document uploaded successfully.")]}


@catch_interruption
def fetch_from_vectordb(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    results = vector_store.similarity_search(query, k=3)
    contents = [doc.page_content for doc in results]
    return {"content": contents}


@catch_interruption
def ai_response(state: AgentState, llm: Any) -> AgentState:
    rag_prompt = f"""Context: {state['content'][-1]}
    
    Based on the above context, answer this question: {state['messages'][-1].content}"""

    if not state["content"]:
        response = llm.invoke(state["messages"])
    else:
        response = llm.invoke(rag_prompt)

    if state.get("verbose", False):
        console.print(state, style=config.get("verbose-color"))

    return {"messages": [AIMessage(content=response.content)]}

from typing import Any, List

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

import src.tools.chroma as chroma_tools
from src.agent.schemas import RouterOutput
from src.agent.states import AgentState
from src.tools.chroma import show_chroma_embedding_fulltext_search_content, show_love
from src.utils.document import process_html
from src.utils.store import vector_store
from src.utils.util import CONTINUE, EXIT, catch_interruption, config, console


@catch_interruption
def tool_decision(state: AgentState, llm: Any) -> AgentState:
    if state.get("decision") in [EXIT, CONTINUE]:
        return state

    llm_with_tools = llm.bind_tools(
        [show_love, show_chroma_embedding_fulltext_search_content]
    )
    result = llm_with_tools.invoke([state["messages"][-1]])

    if state.get("verbose", False):
        console.print(
            f"Tool decision result: {result}", style=config.get("verbose-color")
        )

    return {"messages": result}


@catch_interruption
def router(state: AgentState, llm: Any) -> AgentState:
    # shortcut decision check
    if state.get("decision") in [EXIT, CONTINUE]:
        return state

    # tool call check
    tool_decision_response = state["messages"][-1]
    if tool_decision_response.tool_calls:
        for tool_call in tool_decision_response.tool_calls:
            name = tool_call.get("name")
            _args = tool_call.get("args")
            tool_call_id = tool_call.get("id")

            structured_tool = getattr(chroma_tools, name, None)
            if structured_tool is None:
                raise ValueError(f"Tool {name} not found.")
            tool_result = structured_tool.run(_args)

            state["messages"].append(tool_result)
            state["messages"].append(
                ToolMessage(content=tool_result, name=name, tool_call_id=tool_call_id)
            )
        return {"intent": ["none"], "urls": ["none"]}

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
    # TODO: do I shortcut tool call?
    if not state["content"]:
        response = llm.invoke(state["messages"])
    else:
        rag_prompt = f"""Context: {state['content'][-1]}
        
        Based on the above context, answer this question: {state['messages'][-1].content}"""
        response = llm.invoke(rag_prompt)

    if state.get("verbose", False):
        console.print(state, style=config.get("verbose-color"))

    return {"messages": [AIMessage(content=response.content)]}

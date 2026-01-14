from typing import Any

from langchain.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from rich.prompt import Prompt

from models import RouteQuery
from states import SimpleState
from util import CONTINUE, EXIT, catch_interruption, config, console


@catch_interruption
def initialize_state() -> SimpleState:
    state = SimpleState()
    state["messages"] = []
    state["decision"] = ""
    state["verbose"] = False
    return state


@catch_interruption
def router(state: SimpleState, llm: Any) -> SimpleState:
    print(state.get("decision"))
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
    simple_response = simple_router.invoke({"question": query})
    state["messages"].append(simple_response)

    return state


@catch_interruption
def upload_to_vectordb(state: SimpleState) -> SimpleState:
    # Placeholder for upload logic
    console.print("Uploading document to vector database...", style="yellow")
    state["decision"] = CONTINUE
    return state


@catch_interruption
def fetch_from_vectordb(state: SimpleState) -> SimpleState:
    # Placeholder for fetch logic
    console.print("Fetching information from vector database...", style="yellow")
    state["decision"] = CONTINUE
    return state


@catch_interruption
def ask(state: SimpleState) -> SimpleState:
    state["decision"] = ""
    user_input = Prompt.ask(config["user-prompt-format"])

    if user_input in [":exit", ":q"]:
        state["decision"] = EXIT
        return state

    if user_input == ":verbose":
        state["decision"] = CONTINUE
        state["verbose"] = not state.get("verbose", False)
        console.print(
            f"Verbose mode {'enabled' if state['verbose'] else 'disabled'}.",
            style=config.get("verbose-color"),
        )
        return state

    state["messages"].append(HumanMessage(content=user_input))
    # state["decision"] = "ai_response"
    return state


@catch_interruption
def ai_response(state: SimpleState, llm: Any) -> SimpleState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    state["decision"] = CONTINUE
    console.print(f"{config['ai-prompt-format']}: {response.content}")

    if state.get("verbose", False):
        console.print(state["messages"], style=config.get("verbose-color"))

    return state

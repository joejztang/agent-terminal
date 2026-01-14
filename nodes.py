from typing import Any

from langchain.messages import AIMessage, HumanMessage
from rich.prompt import Prompt

from states import SimpleState
from util import CONTINUE, EXIT, catch_interruption, config, console


@catch_interruption
def initialize_state() -> SimpleState:
    state = SimpleState()
    state["messages"] = []
    state["decision"] = ""
    return state


@catch_interruption
def ask(state: SimpleState) -> SimpleState:
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
    state["decision"] = "ai_response"
    return state


@catch_interruption
def ai_response(state: SimpleState, llm: Any) -> SimpleState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    console.print(f"{config['ai-prompt-format']}: {response.content}")

    if state.get("verbose", False):
        console.print(state["messages"], style=config.get("verbose-color"))

    return state

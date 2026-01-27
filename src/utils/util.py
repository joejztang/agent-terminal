from pathlib import Path

from rich.console import Console
from yaml import safe_load

from src.agent.states import AgentState

file_path = Path(__file__).resolve()
parent_directory = file_path.parent

console = Console()

with open(f"{parent_directory}/config.yaml", "r") as f:
    config = safe_load(f)

EXIT = "exit"
CONTINUE = "continue"
UPLOAD = "upload_to_vectordb"
FETCH = "fetch_from_vectordb"
NO_ACTION = "none"

MAX_DEPTH = 2
HTML_CHUNK_SIZE = 250
HTML_CHUNK_OVERLAP = 50


def initialize_state() -> AgentState:
    state = AgentState()
    state["messages"] = []
    state["decision"] = ""
    state["verbose"] = False
    state["content"] = []
    return state


def check_quit(user_input: str) -> bool:
    return user_input in [":exit", ":q"]


def check_verbose_command(user_input: str) -> bool:
    return user_input == ":verbose"


def toggle_verbose(state: AgentState) -> AgentState:
    state["verbose"] = not state.get("verbose", False)
    return state


def catch_interruption(func):
    def wrapper(*args, **kwargs):
        state = args[0]
        try:
            if state.get("decision") is EXIT:
                return state
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            state["decision"] = EXIT
            return state

    return wrapper

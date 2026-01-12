from rich.console import Console
from yaml import safe_load

console = Console()

with open("config.yaml", "r") as f:
    config = safe_load(f)

EXIT = "exit"
CONTINUE = "continue"


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

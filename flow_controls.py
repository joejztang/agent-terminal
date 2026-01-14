from states import SimpleState
from util import CONTINUE, EXIT, catch_interruption


def simple_after_ask(state: SimpleState) -> str:
    if state.get("decision") is EXIT:
        return EXIT
    elif state.get("decision") is CONTINUE:
        return CONTINUE
    else:
        return "ai_response"


def simple_after_ai_response(state: SimpleState) -> str:
    if state.get("decision") is EXIT:
        return EXIT
    elif state.get("decision") is CONTINUE:
        return CONTINUE
    else:
        return "ask"

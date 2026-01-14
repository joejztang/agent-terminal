from states import SimpleState
from util import CONTINUE, EXIT, FETCH, NO_ACTION, UPLOAD, catch_interruption


def after_router(state: SimpleState) -> str:
    if state.get("decision") is EXIT:
        return EXIT
    if state.get("decision") is CONTINUE:
        return CONTINUE
    action = state["messages"][-1].content.strip()
    if action in [UPLOAD, FETCH, NO_ACTION]:
        return action
    return NO_ACTION


def simple_after_ask(state: SimpleState) -> str:
    if state.get("decision") is EXIT:
        return EXIT
    elif state.get("decision") is CONTINUE:
        return CONTINUE
    else:
        return "ai_response"


def simple_after_control(state: SimpleState) -> str:
    if state.get("decision") in [EXIT, CONTINUE]:
        return state.get("decision")
    else:
        return "ask"

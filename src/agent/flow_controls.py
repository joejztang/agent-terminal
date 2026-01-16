from src.agent.states import AgentState
from src.utils.util import CONTINUE, EXIT, FETCH, NO_ACTION, UPLOAD, catch_interruption


def after_router(state: AgentState) -> str:
    if state.get("decision") is EXIT:
        return EXIT
    action = state["messages"][-1].content.strip()
    if action in [UPLOAD, FETCH, NO_ACTION]:
        return action
    return NO_ACTION


def simple_after_ask(state: AgentState) -> str:
    if state.get("decision") is EXIT:
        return EXIT
    elif state.get("decision") is CONTINUE:
        return CONTINUE
    else:
        return "ai_response"


def simple_after_control(state: AgentState) -> str:
    if state.get("decision") in [EXIT, CONTINUE]:
        return state.get("decision")
    else:
        return "ask"

from src.agent.states import AgentState
from src.utils.util import CONTINUE, EXIT, FETCH, NO_ACTION, UPLOAD, catch_interruption


def after_router(state: AgentState) -> str:
    if state.get("decision") is EXIT:
        return EXIT
    return state["intent"][-1]


def after_tool_decision(state: AgentState) -> str:
    if state.get("decision") in [EXIT, CONTINUE]:
        return state.get("decision")
    else:
        return EXIT


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

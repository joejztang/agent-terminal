from art import tprint
from langchain_core.messages import AIMessage, HumanMessage
from rich.prompt import Prompt

from src.agent.graphs import agent_graph
from src.agent.llms import qwen3_8b
from src.utils.util import (
    check_quit,
    check_verbose_command,
    config,
    console,
    initialize_state,
    toggle_verbose,
)


def main():
    llm = qwen3_8b
    graph = agent_graph(llm)

    print(graph.get_graph().draw_ascii())

    state = initialize_state()
    while True:
        user_input = Prompt.ask(config["user-prompt-format"])

        if check_quit(user_input):
            break

        if check_verbose_command(user_input):
            state = toggle_verbose(state)
            console.print(
                f"Verbose mode {'enabled' if state['verbose'] else 'disabled'}.",
                style=config.get("verbose-color"),
            )
            continue

        state["messages"].append(HumanMessage(content=user_input))
        all_responses = graph.invoke(state)

        state = all_responses

        response = []
        for message in reversed(all_responses["messages"]):
            if not isinstance(message, HumanMessage):
                response.append(message)
            else:
                break
        # TODO: make it pretty
        console.print(f"{config['ai-prompt-format']}: {response}")


if __name__ == "__main__":
    tprint("agent-terminal")
    main()

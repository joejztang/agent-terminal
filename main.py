from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from rich.prompt import Prompt

from graphs import agent_graph
from llms import ministral_3_14b
from util import (
    check_quit,
    check_verbose_command,
    config,
    console,
    initialize_state,
    toggle_verbose,
)


def main():
    llm = ministral_3_14b
    graph = agent_graph(llm)

    print(graph.get_graph().draw_ascii())
    # with open("graph.png", "wb") as f:
    #     f.write(
    #         graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
    #     )
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

        response = ""
        for message in reversed(all_responses["messages"]):
            if isinstance(message, AIMessage):
                response = message
                break
        console.print(f"{config['ai-prompt-format']}: {response.content}")


if __name__ == "__main__":
    main()

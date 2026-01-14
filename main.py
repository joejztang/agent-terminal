from langchain_core.runnables.graph import MermaidDrawMethod

from graphs import simple_chat
from llms import ministral_3_14b


def main():
    llm = ministral_3_14b
    graph = simple_chat(llm)
    graph = graph.compile()

    # print(graph.get_graph().draw_ascii())

    graph.invoke({})


if __name__ == "__main__":
    main()

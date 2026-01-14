from langchain_core.runnables.graph import MermaidDrawMethod

from graphs import simple_chat
from llms import ministral_3_14b


def main():
    llm = ministral_3_14b
    graph = simple_chat(llm)
    graph = graph.compile()

    print(graph.get_graph().draw_ascii())
    # with open("graph.png", "wb") as f:
    #     f.write(
    #         graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
    #     )

    graph.invoke({}, {"recursion_limit": 1000})


if __name__ == "__main__":
    main()

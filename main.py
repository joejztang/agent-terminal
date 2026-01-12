from langchain_ollama import ChatOllama

from graphs import simple_chat
from llms import ministral_3_14b


def main():
    llm = ministral_3_14b
    graph = simple_chat(llm)
    graph = graph.compile()

    # with open("graph.png", "wb") as f:
    #     f.write(graph.get_graph().draw_mermaid_png())

    graph.invoke({})


if __name__ == "__main__":
    main()

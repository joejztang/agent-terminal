from langchain_chroma import Chroma

from src.agent.llms import qwen3_embeddings_function

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=qwen3_embeddings_function,
    persist_directory="./chroma_langchain_db",
)

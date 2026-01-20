from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_ollama import ChatOllama

ministral_3_14b = ChatOllama(model="ministral-3:14b", temperature=0)

qwen3_8b = ChatOllama(model="qwen3:8b", temperature=0)

qwen3_embeddings_function = OllamaEmbeddingFunction(model_name="qwen3-embeddings:0.6b")

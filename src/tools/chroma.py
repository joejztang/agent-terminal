from typing import List

from langchain_core.tools import tool

from src.utils.store import vector_store


@tool
def show_love() -> str:
    """Give response expressing love."""
    return "I love you gatja;ljfojnasijeo!"


@tool(
    name_or_callable="show_fulltexts",
    description="Show all fulltexts stored in the Chroma vector database.",
)
def show_chroma_embedding_fulltext_search_content() -> List[str]:
    full = vector_store._collection.get()
    return full.get("documents", [])


@tool
def delete_before_date(date_str: str) -> str:
    """Delete all documents added before the given date string (YYYY-MM-DD)."""
    vector_store.delete(filter={"$lt": {"_ts": date_str}})
    return f"Deleted documents before {date_str}."

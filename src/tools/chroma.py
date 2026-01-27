from datetime import datetime
from typing import List

from langchain_core.tools import tool

from src.utils.store import vector_store


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
    time_obj = datetime.strptime(date_str, "%Y-%m-%d")  # validate format
    vector_store.delete(where={"created_at": {"$lt": int(time_obj.timestamp())}})
    return f"Deleted documents before {date_str}."

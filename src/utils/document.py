import re
from typing import List

from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.util import MAX_DEPTH


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # remove noise
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    # better text extraction
    text = soup.get_text(separator="\n", strip=True)
    return re.sub(r"\n\n+", "\n\n", text).strip()


def web_scrape(url: str) -> List[Document]:
    """Fetch and return the text content of a web page."""
    loader = RecursiveUrlLoader(
        url, max_depth=MAX_DEPTH, extractor=bs4_extractor, use_async=True
    )
    docs = loader.load()
    return docs


def process_html(url) -> List[Document]:
    docs = web_scrape(url)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    return texts

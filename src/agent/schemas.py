from typing import List

from pydantic import BaseModel


class RouterOutput(BaseModel):
    query: List[str]
    intent: List[str]
    urls: List[str]

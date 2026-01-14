from typing import Literal

from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    action: Literal["upload_to_vectordb", "fetch_from_vectordb", "none"] = Field(
        description="Given a question choose to route to an action."
    )

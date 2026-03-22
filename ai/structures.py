from pydantic import BaseModel, Field


class UserMemory(BaseModel):
    facts: list[str] = Field(
        default_factory=list, description="A list of facts about the user"
    )


__all__ = [
    "UserMemory",
]

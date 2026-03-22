from pydantic import BaseModel, Field


class UserMemory(BaseModel):
    facts: list[str] = Field(
        default_factory=list, description="A list of facts about the user"
    )


class MemoryUpdate(BaseModel):
    should_update: bool = Field(
        description="True if the user is correcting or updating an existing fact"
    )
    key: str = Field(default="", description="The key of the memory to update")
    updated_fact: str = Field(default="", description="The corrected fact text")


class MemoryDelete(BaseModel):
    should_delete: bool = Field(
        description="True if the user wants to forget or remove a memory"
    )
    keys: list[str] = Field(
        default_factory=list, description="Keys of the memories to delete"
    )


__all__ = [
    "MemoryDelete",
    "MemoryUpdate",
    "UserMemory",
]

from typing import Annotated, TypedDict

from langchain.messages import AnyMessage

MAX_MESSAGES = 3


def _keep_last(existing: list, new: list) -> list:
    return (existing + new)[-MAX_MESSAGES:]


class ChatBotState(TypedDict):
    memory_context: str
    messages: Annotated[list[AnyMessage], _keep_last]
    user_query: str


__all__ = [
    "ChatBotState",
]

from operator import add
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage


class ChatBotState(TypedDict):
    memory_context: str
    messages: Annotated[list[AnyMessage], add]
    user_query: str


__all__ = [
    "ChatBotState",
]

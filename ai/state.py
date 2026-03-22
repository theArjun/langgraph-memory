from operator import add
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage


class ChatBotState(TypedDict):
    memory_context: str
    messages: Annotated[list[AnyMessage], add]


__all__ = [
    "ChatBotState",
]

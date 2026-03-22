from langchain_core.messages import AIMessage

from ..state import ChatBotState


class ChatbotRouter:
    TOOLS = "tools"
    CLEAR_CHECKPOINTS = "clear_checkpoints"

    def __call__(self, state: ChatBotState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return self.TOOLS
        return self.CLEAR_CHECKPOINTS


chatbot_router = ChatbotRouter()

__all__ = ["chatbot_router", "ChatbotRouter"]

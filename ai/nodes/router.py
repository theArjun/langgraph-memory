from langchain_core.messages import AIMessage
from langgraph.graph import END

from ..state import ChatBotState


class ChatbotRouter:
    TOOLS = "tools"
    END = END

    def __call__(self, state: ChatBotState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return self.TOOLS
        return self.END


chatbot_router = ChatbotRouter()

__all__ = ["chatbot_router", "ChatbotRouter"]

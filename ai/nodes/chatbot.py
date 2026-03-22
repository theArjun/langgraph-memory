from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..llm import llm
from ..logger import get_logger
from ..state import ChatBotState

logger = get_logger(__name__)

MAX_TOKENS = 100_000


def chatbot(state: ChatBotState, config: RunnableConfig):
    config["configurable"]["user_id"]
    memory_context = state.get("memory_context", {})
    user_query = state.get("user_query")

    if memory_context:
        system_prompt = f"""You're a helpful assistant with memory of past conversations.
        Here is the conversation history:
        {memory_context}

        User this information to personalize your response. Be natural and conversational.
        """
    else:
        system_prompt = """You're a helpful assistant. This is your first conversation with this user. Be natural and conversational.
        """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    logger.info(
        "Invoking LLM with memory_context=%s", "present" if memory_context else "absent"
    )
    response = llm.invoke(messages)
    return {
        "messages": [response],
    }

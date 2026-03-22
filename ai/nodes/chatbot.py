from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..llm import llm
from ..logger import get_logger
from ..prompts import load_prompt
from ..state import ChatBotState

logger = get_logger(__name__)

MAX_TOKENS = 100_000


def chatbot(state: ChatBotState, config: RunnableConfig):
    config["configurable"]["user_id"]
    memory_context = state.get("memory_context", "")
    user_query = state.get("user_query")

    system_prompt = load_prompt("chatbot", memory_context=memory_context)

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

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..llm import llm
from ..logger import get_logger
from ..prompts import load_prompt
from ..state import ChatBotState
from ..tools import memory_tools

logger = get_logger(__name__)

_llm_with_tools = llm.bind_tools(memory_tools)


def chatbot(state: ChatBotState, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    user_query = state.get("user_query")

    system_prompt = load_prompt("chatbot")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    # On re-entry after tool calls, append tool call/result messages
    # so the LLM can see what it called and produce a final response.
    for msg in state.get("messages", []):
        if (hasattr(msg, "tool_calls") and msg.tool_calls) or msg.type == "tool":
            messages.append(msg)

    logger.info("Invoking LLM for user=%s", user_id)
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response]}

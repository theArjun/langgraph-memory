from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..llm import llm
from ..logger import get_logger
from ..prompts import load_prompt
from ..state import ChatBotState
from ..store import store_manager
from ..structures import UserMemory

logger = get_logger(__name__)


def extract_and_save(state: ChatBotState, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]

    user_query = state.get("user_query")
    ai_responses = [m for m in state["messages"] if m.type == "ai"]

    if not user_query or not ai_responses:
        return state

    last_ai_response = ai_responses[-1].content

    prompt = load_prompt(
        "extract_and_save", user_query=user_query, ai_response=last_ai_response
    )
    messages = [SystemMessage(content=prompt["system"]), HumanMessage(content=prompt["user"])]
    result = llm.with_structured_output(UserMemory).invoke(messages)

    stored_count = sum(
        store_manager.save(user_id, fact) for fact in result.facts if fact
    )

    logger.info(
        "Extracted %d facts, stored %d new for user=%s",
        len(result.facts),
        stored_count,
        user_id,
    )

    return state

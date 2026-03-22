from langchain_core.runnables import RunnableConfig

from ..llm import llm
from ..logger import get_logger
from ..prompts import load_prompt
from ..state import ChatBotState
from ..store import store_manager
from ..structures import MemoryUpdate

logger = get_logger(__name__)


def update_memory(state: ChatBotState, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    user_query = state.get("user_query")

    memories = store_manager.search(user_id, query=user_query)
    if not memories:
        return state

    prompt = load_prompt("update_memory", user_query=user_query, memories=memories)
    messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
    result: MemoryUpdate = llm.with_structured_output(MemoryUpdate).invoke(messages)

    if result.should_update and result.key and result.updated_fact:
        store_manager.update(user_id, result.key, result.updated_fact)
        logger.info("Updated memory key=%s for user=%s", result.key, user_id)

    return state

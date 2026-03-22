from langchain_core.runnables import RunnableConfig

from ..llm import llm
from ..logger import get_logger
from ..prompts import load_prompt
from ..state import ChatBotState
from ..store import store_manager
from ..structures import MemoryDelete

logger = get_logger(__name__)


def delete_memory(state: ChatBotState, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    user_query = state.get("user_query")

    memories = store_manager.search(user_id, query=user_query)
    if not memories:
        return state

    prompt = load_prompt("delete_memory", user_query=user_query, memories=memories)
    messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
    result: MemoryDelete = llm.with_structured_output(MemoryDelete).invoke(messages)

    if result.should_delete and result.keys:
        for key in result.keys:
            store_manager.delete(user_id, key)
        logger.info("Deleted %d memories for user=%s", len(result.keys), user_id)

    return state

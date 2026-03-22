from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .logger import get_logger
from .store import store_manager

logger = get_logger(__name__)


@tool
def retrieve_memories(query: str, config: RunnableConfig) -> str:
    """Retrieve stored memories about the user relevant to a query.
    Call this at the start of a conversation or when past context would help personalize the response."""
    user_id = config["configurable"]["user_id"]
    memories = store_manager.search(user_id, query=query)
    if not memories:
        return "No memories found."
    lines = [f"[key={m.key}] {m.value.get('text', '')}" for m in memories]
    logger.info("Tool: retrieved %d memories for user=%s", len(memories), user_id)
    return "\n".join(lines)


@tool
def update_memory(updated_fact: str, config: RunnableConfig) -> str:
    """Update an existing memory with a corrected fact.
    Use when the user corrects or changes previously stored information."""
    user_id = config["configurable"]["user_id"]
    store_manager.save(user_id, updated_fact)
    logger.info("Tool: update_memory for user=%s fact=%s", user_id, updated_fact)
    return f"Updated: {updated_fact}"


@tool
def delete_memory(key: str, config: RunnableConfig) -> str:
    """Delete a stored memory.
    Use when the user explicitly asks to forget or remove specific information.
    'key' must be the exact memory key shown in the conversation context."""
    user_id = config["configurable"]["user_id"]
    store_manager.delete(user_id, key)
    logger.info("Tool: deleted memory key=%s for user=%s", key, user_id)
    return f"Deleted memory '{key}'"


@tool
def save_memory(fact: str, config: RunnableConfig) -> str:
    """Save a new fact about the user to memory.
    Call this when the user shares personal information worth remembering for future conversations."""
    user_id = config["configurable"]["user_id"]
    saved = store_manager.save(user_id, fact)
    logger.info("Tool: save_memory for user=%s saved=%s fact=%s", user_id, saved, fact)
    return f"Saved: {fact}" if saved else f"Already known: {fact}"


memory_tools = [retrieve_memories, save_memory, update_memory, delete_memory]

__all__ = ["memory_tools"]

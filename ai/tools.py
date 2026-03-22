from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .logger import get_logger
from .store import store_manager

logger = get_logger(__name__)


@tool
def update_memory(key: str, updated_fact: str, config: RunnableConfig) -> str:
    """Update an existing memory with a corrected fact.
    Use when the user corrects or changes previously stored information.
    'key' must be the exact memory key shown in the conversation context."""
    user_id = config["configurable"]["user_id"]
    store_manager.update(user_id, key, updated_fact)
    logger.info("Tool: updated memory key=%s for user=%s", key, user_id)
    return f"Updated memory '{key}' to: {updated_fact}"


@tool
def delete_memory(key: str, config: RunnableConfig) -> str:
    """Delete a stored memory.
    Use when the user explicitly asks to forget or remove specific information.
    'key' must be the exact memory key shown in the conversation context."""
    user_id = config["configurable"]["user_id"]
    store_manager.delete(user_id, key)
    logger.info("Tool: deleted memory key=%s for user=%s", key, user_id)
    return f"Deleted memory '{key}'"


memory_tools = [update_memory, delete_memory]

__all__ = ["memory_tools"]

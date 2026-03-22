from langchain_core.runnables import RunnableConfig

from ..state import ChatBotState
from ..store import store_manager


def retrieve_memories(state: ChatBotState, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]

    memories = store_manager.search(
        user_id, query="What are the facts about this user?"
    )

    memory_context = ""
    if memories:
        memory_texts = [
            f"{i}. {m.value.get('text', '')}" for i, m in enumerate(memories, start=1)
        ]
        memory_context = "\n".join(memory_texts)

    return {"memory_context": memory_context}

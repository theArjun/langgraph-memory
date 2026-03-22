from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from ..state import ChatBotState


def retrieve_memories(state: ChatBotState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    user_memories_namespace = (user_id, "memories")

    memories = store.search(
        user_memories_namespace, query="What are the facts about this user ?"
    )

    memory_context = ""

    if memories:
        memory_texts = []
        for i, memory in enumerate(memories, start=1):
            text = memory.value.get("text", "")
            memory_texts.append(f"{i}. {text}")
        memory_context = "\n".join(memory_texts)

    return {"memory_context": memory_context}

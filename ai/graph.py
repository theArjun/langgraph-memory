from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import chatbot, extract_and_save, retrieve_memories
from .state import ChatBotState
from .store import store_manager


class GraphNodes:
    EXTRACT_AND_SAVE = "extract_and_save"
    CHATBOT = "chatbot"
    RETRIEVE_MEMORIES = "retrieve_memories"


builder = StateGraph(ChatBotState)

builder.add_node(GraphNodes.EXTRACT_AND_SAVE, extract_and_save)
builder.add_node(GraphNodes.CHATBOT, chatbot)
builder.add_node(GraphNodes.RETRIEVE_MEMORIES, retrieve_memories)

builder.add_edge(START, GraphNodes.RETRIEVE_MEMORIES)
builder.add_edge(GraphNodes.RETRIEVE_MEMORIES, GraphNodes.CHATBOT)
builder.add_edge(GraphNodes.CHATBOT, GraphNodes.EXTRACT_AND_SAVE)
builder.add_edge(GraphNodes.EXTRACT_AND_SAVE, END)

checkpointer = MemorySaver()


graph = builder.compile(checkpointer=checkpointer, store=store_manager._store)


__all__ = [
    "graph",
]

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from .logger import get_logger
from .nodes import (chatbot, clear_checkpoints, extract_and_save,
                    retrieve_memories)
from .state import ChatBotState
from .store import checkpointer, store_manager
from .tools import memory_tools

logger = get_logger(__name__)


class GraphNodes:
    RETRIEVE_MEMORIES = "retrieve_memories"
    CHATBOT = "chatbot"
    TOOLS = "tools"
    EXTRACT_AND_SAVE = "extract_and_save"
    CLEAR_CHECKPOINTS = "clear_checkpoints"


def _should_use_tools(state: ChatBotState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return GraphNodes.TOOLS
    return GraphNodes.EXTRACT_AND_SAVE


def _build_graph():
    logger.info("Building graph")
    builder = StateGraph(ChatBotState)

    builder.add_node(GraphNodes.RETRIEVE_MEMORIES, retrieve_memories)
    builder.add_node(GraphNodes.CHATBOT, chatbot)
    builder.add_node(GraphNodes.TOOLS, ToolNode(memory_tools))
    builder.add_node(GraphNodes.EXTRACT_AND_SAVE, extract_and_save)
    builder.add_node(GraphNodes.CLEAR_CHECKPOINTS, clear_checkpoints)

    builder.add_edge(START, GraphNodes.RETRIEVE_MEMORIES)
    builder.add_edge(GraphNodes.RETRIEVE_MEMORIES, GraphNodes.CHATBOT)
    builder.add_conditional_edges(GraphNodes.CHATBOT, _should_use_tools)
    builder.add_edge(GraphNodes.TOOLS, GraphNodes.CHATBOT)
    builder.add_edge(GraphNodes.EXTRACT_AND_SAVE, GraphNodes.CLEAR_CHECKPOINTS)
    builder.add_edge(GraphNodes.CLEAR_CHECKPOINTS, END)

    return builder.compile(checkpointer=checkpointer, store=store_manager._store)


class GraphManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._graph = _build_graph()
        return cls._instance

    @traceable
    def invoke(self, user_id: str, thread_id: str, message: str) -> str:
        logger.info("Invoking graph for user=%s thread=%s", user_id, thread_id)
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        result = self._graph.invoke(
            {"user_query": message},
            config=config,
        )
        return result["messages"][-1].content


graph_manager = GraphManager()

__all__ = [
    "graph_manager",
]

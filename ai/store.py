from langgraph.store.memory import InMemoryStore

from .embeddings import embeddings

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["text", "$"],
    }
)

__all__ = [
    "store",
]

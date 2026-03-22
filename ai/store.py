from datetime import datetime

from langgraph.store.memory import InMemoryStore

from .embeddings import embeddings

SIMILARITY_THRESHOLD = 0.90


class StoreManager:
    def __init__(self, store: InMemoryStore):
        self._store = store

    def _namespace(self, user_id: str) -> tuple:
        return (user_id, "memories")

    def search(self, user_id: str, query: str, limit: int = 10) -> list:
        return self._store.search(self._namespace(user_id), query=query, limit=limit)

    def save(self, user_id: str, fact: str) -> bool:
        similar = self._store.search(self._namespace(user_id), query=fact, limit=1)
        if similar and similar[0].score >= SIMILARITY_THRESHOLD:
            return False

        now = datetime.now()
        self._store.put(
            namespace=self._namespace(user_id),
            key=f"memory_{now.strftime('%Y%m%d_T%H%M%S')}",
            value={
                "text": fact,
                "timestamp": now.isoformat(),
                "source": "conversation",
            },
        )
        return True


_store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["text", "$"],
    }
)

store_manager = StoreManager(_store)

__all__ = [
    "store_manager",
]

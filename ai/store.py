from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from .config import get_database_url
from .embeddings import embeddings
from .logger import get_logger

from psycopg import connect
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.90


class StoreManager:
    def __init__(self, store: InMemoryStore):
        self._store = store

    def _namespace(self, user_id: str) -> tuple:
        return (user_id, "memories")

    def search(self, user_id: str, query: str, limit: int = 10) -> list:
        return self._store.search(self._namespace(user_id), query=query, limit=limit)

    def save(self, user_id: str, fact: str) -> bool:
        if not fact or not user_id:
            return False

        similar = self._store.search(self._namespace(user_id), query=fact, limit=1)
        if similar and similar[0].score >= SIMILARITY_THRESHOLD:
            logger.debug(
                "Skipping duplicate fact for %s (score=%.2f): %s",
                user_id,
                similar[0].score,
                fact,
            )
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
        logger.info("Saved new fact for %s: %s", user_id, fact)
        return True


_store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["text", "$"],
    }
)

store_manager = StoreManager(_store)


def _build_checkpointer():
    db_url = get_database_url()
    if not db_url:
        logger.info("No DATABASE_URL set, defaulting to MemorySaver (in-memory)")
        return MemorySaver()

    logger.info("DATABASE_URL detected, attempting PostgresSaver connection")
    try:
        conn = connect(db_url, autocommit=True, row_factory=dict_row)
        saver = PostgresSaver(conn)
        saver.setup()
        logger.info("Checkpointer: PostgresSaver (connected to Postgres)")
        return saver
    except Exception as e:
        logger.warning("PostgresSaver init failed (%s), falling back to MemorySaver", e)
        return MemorySaver()


checkpointer = _build_checkpointer()

__all__ = [
    "store_manager",
    "checkpointer",
]

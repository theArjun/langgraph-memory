from datetime import datetime
from typing import Union

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from psycopg import connect
from psycopg.rows import dict_row

from .config import get_database_url
from .embeddings import embeddings
from .logger import get_logger

logger = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.90


_INDEX_CONFIG = {
    "embed": embeddings,
    "dims": 1536,
    "fields": ["text", "$"],
}


class StoreManager:
    def __init__(
        self, store: Union[InMemoryStore, PostgresStore], has_index: bool = True
    ):
        self._store = store
        self._has_index = has_index

    def _namespace(self, user_id: str) -> tuple:
        return (user_id, "memories")

    def search(self, user_id: str, query: str, limit: int = 10) -> list:
        if not self._has_index:
            return []
        return self._store.search(self._namespace(user_id), query=query, limit=limit)

    def update(self, user_id: str, key: str, new_fact: str) -> None:
        now = datetime.now()
        self._store.put(
            namespace=self._namespace(user_id),
            key=key,
            value={
                "text": new_fact,
                "timestamp": now.isoformat(),
                "source": "conversation",
            },
        )
        logger.info("Updated memory %s for %s: %s", key, user_id, new_fact)

    def delete(self, user_id: str, key: str) -> None:
        self._store.delete(namespace=self._namespace(user_id), key=key)
        logger.info("Deleted memory %s for %s", key, user_id)

    def save(self, user_id: str, fact: str) -> bool:
        if not fact or not user_id:
            return False

        if self._has_index:
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


def _build_store() -> tuple[Union[InMemoryStore, PostgresStore], bool]:
    db_url = get_database_url()
    if not db_url:
        logger.info("No DATABASE_URL set, defaulting to InMemoryStore")
        return InMemoryStore(index=_INDEX_CONFIG), True

    logger.info("DATABASE_URL detected, attempting PostgresStore connection")
    conn = connect(db_url, autocommit=True, row_factory=dict_row)

    # Try with vector index first (requires pgvector)
    try:
        store = PostgresStore(conn, index=_INDEX_CONFIG)
        store.setup()
        logger.info("Store: PostgresStore with vector index (pgvector enabled)")
        return store, True
    except Exception as e:
        logger.warning(
            "PostgresStore with index failed (%s), retrying without vector index", e
        )

    # Retry without index (pgvector not available)
    try:
        conn = connect(db_url, autocommit=True, row_factory=dict_row)
        store = PostgresStore(conn)
        store.setup()
        logger.warning("Store: PostgresStore without vector index (no semantic search)")
        return store, False
    except Exception as e:
        logger.warning(
            "PostgresStore init failed (%s), falling back to InMemoryStore", e
        )
        return InMemoryStore(index=_INDEX_CONFIG), True


def _build_checkpointer() -> Union[PostgresSaver, InMemorySaver]:
    db_url = get_database_url()
    if not db_url:
        logger.info("No DATABASE_URL set, defaulting to InMemorySaver (in-memory)")
        return InMemorySaver()

    logger.info("DATABASE_URL detected, attempting PostgresSaver connection")
    try:
        conn = connect(db_url, autocommit=True, row_factory=dict_row)
        saver = PostgresSaver(conn)
        saver.setup()
        logger.info("Checkpointer: PostgresSaver (connected to Postgres)")
        return saver
    except Exception as e:
        logger.warning(
            "PostgresSaver init failed (%s), falling back to InMemorySaver", e
        )
        return InMemorySaver()


def delete_thread(thread_id: str) -> None:
    db_url = get_database_url()
    if not db_url:
        return
    with connect(db_url, autocommit=True, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,)
            )
            cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
    logger.info("Deleted checkpoint for thread_id=%s", thread_id)


_store, _has_index = _build_store()
store_manager = StoreManager(_store, _has_index)
checkpointer = _build_checkpointer()

__all__ = [
    "checkpointer",
    "delete_thread",
    "store_manager",
]

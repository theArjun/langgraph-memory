from langchain_core.runnables import RunnableConfig
from psycopg import connect
from psycopg.rows import dict_row

from ..config import get_database_url
from ..logger import get_logger
from ..state import ChatBotState

logger = get_logger(__name__)


def clear_checkpoints(state: ChatBotState, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    db_url = get_database_url()

    if not db_url:
        return state

    with connect(db_url, autocommit=True, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,)
            )
            cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,)
            )

    logger.info("Cleared checkpoints for thread_id=%s", thread_id)
    return state

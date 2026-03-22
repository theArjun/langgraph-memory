# LangGraph Memory

A conversational AI chatbot that extracts and persists facts about users across interactions using LangGraph with PostgreSQL-backed checkpointing and memory store.

## How it works

Each conversation runs through a three-node graph:

![Graph](graph.png)

1. **retrieve_memories** — searches the store for previously saved facts about the user using semantic similarity
2. **chatbot** — calls GPT-4o with the retrieved memory context to produce a personalized response
3. **extract_and_save** — uses structured output to extract new facts from the last exchange and saves them to the store
4. **clear_checkpoints** — deletes the checkpoint rows for the current thread after a successful run, keeping the checkpointer table lean

Facts are stored under a per-user namespace `(user_id, "memories")` and embedded with `text-embedding-3-small` for semantic retrieval. Before saving, each new fact is checked against existing memories using a similarity threshold (`0.90`) to avoid storing duplicates.

## Project structure

```
ai/
  graph.py            # GraphManager singleton — builds and compiles the graph
  llm.py              # ChatOpenAI instance (GPT-4o)
  embeddings.py       # OpenAI embeddings instance (text-embedding-3-small)
  store.py            # StoreManager + PostgresStore/InMemoryStore + PostgresSaver/MemorySaver
  state.py            # ChatBotState TypedDict
  structures.py       # UserMemory Pydantic model for structured fact extraction
  config.py           # Loads .env with override=True
  models.py           # LLM and embedding model name constants
  logger.py           # Shared get_logger() factory
  nodes/
    retrieve_memories.py  # Searches store for user facts via StoreManager
    chatbot.py            # Invokes LLM with memory context
    extract_and_save.py   # Extracts and stores new facts via StoreManager
    clear_checkpoints.py  # Deletes checkpoint rows for the thread after each successful run
main.py               # Entry point
visualize_graph.py    # Renders the graph as graph.png
```

## Setup

1. Clone the repo and install dependencies with `uv`:

```bash
uv sync
```

2. Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

3. Run:

```bash
python main.py
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for the LLM (GPT-4o) and embeddings (text-embedding-3-small) |
| `DATABASE_URL` | No | Postgres connection string. Falls back to in-memory if unset. |
| `LANGSMITH_TRACING` | No | Set to `true` to enable LangSmith tracing |
| `LANGSMITH_ENDPOINT` | No | LangSmith API endpoint |
| `LANGSMITH_API_KEY` | No | LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

> `.env` values take precedence over shell environment variables (`override=True`).

## Persistence

Both the **checkpointer** (conversation history) and **store** (extracted memories) use PostgreSQL when `DATABASE_URL` is set, and fall back to in-memory equivalents otherwise.

```env
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

### Checkpointer

| Condition | Backend |
|---|---|
| `DATABASE_URL` set and reachable | `PostgresSaver` |
| Otherwise | `MemorySaver` (lost on restart) |

### Memory store

| Condition | Backend |
|---|---|
| `DATABASE_URL` set, pgvector available | `PostgresStore` with vector index (full semantic search) |
| `DATABASE_URL` set, no pgvector | `PostgresStore` without index (persistent, no semantic search) |
| No `DATABASE_URL` | `InMemoryStore` with vector index (lost on restart) |

Schema migrations are run automatically on startup (`setup()` on both the saver and the store).

#### pgvector

The vector index in `PostgresStore` requires the [pgvector](https://github.com/pgvector/pgvector) extension. If it is not installed, the store falls back to `PostgresStore` without an index — facts are still persisted but semantic deduplication and retrieval are disabled.

To install pgvector on Postgres:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Memory deduplication

`StoreManager.save()` performs a semantic similarity search before writing each new fact. If an existing memory scores `>= 0.90` against the candidate fact, the write is skipped. This requires the vector index (pgvector or InMemoryStore) — when running without an index, deduplication is skipped and all extracted facts are written.

Duplicate skips are logged at `DEBUG` level. To see them, set `LOG_LEVEL=DEBUG` in your environment or adjust the level in `ai/logger.py`.

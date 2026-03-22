# LangGraph Memory

A conversational AI chatbot that extracts and persists facts about users across interactions using LangGraph's in-memory store with semantic search.

## How it works

Each conversation runs through a three-node graph:

```
retrieve_memories → chatbot → extract_and_save
```

1. **retrieve_memories** — searches the store for previously saved facts about the user using semantic similarity
2. **chatbot** — calls GPT-4o with the retrieved memory context to produce a personalized response
3. **extract_and_save** — uses structured output to extract new facts from the conversation and saves them to the store

Facts are stored under a per-user namespace `(user_id, "memories")` and embedded with `text-embedding-3-small` for semantic retrieval. Before saving, each new fact is checked against existing memories using a similarity threshold (`0.90`) to avoid storing duplicates.

## Project structure

```
ai/
  graph.py            # GraphManager singleton — builds and compiles the graph
  llm.py              # ChatOpenAI instance
  embeddings.py       # OpenAI embeddings instance (text-embedding-3-small)
  store.py            # StoreManager (InMemoryStore) + checkpointer (PostgresSaver or MemorySaver)
  state.py            # ChatBotState TypedDict
  structures.py       # UserMemory Pydantic model
  config.py           # Loads .env with override=True
  models.py           # LLM and embedding model name constants
  logger.py           # Shared get_logger() factory
  nodes/
    retrieve_memories.py  # Searches store for user facts via StoreManager
    chatbot.py            # Invokes LLM with memory context
    extract_and_save.py   # Extracts and stores new facts via StoreManager
main.py               # Entry point
```

## Setup

1. Clone the repo and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or with `uv`:

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
| `DATABASE_URL` | No | Postgres connection string for persistent checkpointing. Falls back to in-memory if unset. |

> `.env` values take precedence over shell environment variables (`override=True`).

## Checkpointing

By default the graph uses `MemorySaver` (in-memory, lost on restart). To persist conversation history across runs, set `DATABASE_URL` in `.env` and install the Postgres extras:

```bash
pip install langgraph-checkpoint-postgres "psycopg[binary]"
```

```env
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

`store.py` will automatically connect, run schema migrations (`saver.setup()`), and use `PostgresSaver`. If the connection fails, it logs a warning and falls back to `MemorySaver`.

## Memory deduplication

`StoreManager.save()` performs a semantic similarity search before writing each new fact. If an existing memory scores `>= 0.90` against the candidate fact, the write is skipped. Duplicate skips are logged at `DEBUG` level — set `LOG_LEVEL=DEBUG` or change the level in `ai/logger.py` to see them.

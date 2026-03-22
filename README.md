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

Facts are stored under a per-user namespace `(user_id, "memories")` and embedded with `text-embedding-3-small` for semantic retrieval.

## Project structure

```
ai/
  graph.py            # Graph definition, store, and checkpointer setup
  llm.py              # ChatOpenAI instance
  state.py            # ChatBotState TypedDict
  structures.py       # UserMemory Pydantic model
  config.py           # Loads .env with override=True
  models.py           # LLM and embedding model name constants
  nodes/
    retrieve_memories.py  # Searches store for user facts
    chatbot.py            # Invokes LLM with memory context
    extract_and_save.py   # Extracts and stores new facts
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

2. Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
```

3. Run:

```bash
python main.py
```

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key used for both the LLM (GPT-4o) and embeddings (text-embedding-3-small) |

> Note: `.env` values take precedence over shell environment variables (`override=True`).

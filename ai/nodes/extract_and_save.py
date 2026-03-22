from langchain_core.runnables import RunnableConfig

from ..llm import llm
from ..state import ChatBotState
from ..store import store_manager
from ..structures import UserMemory


def extract_and_save(state: ChatBotState, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]

    if len(state["messages"]) < 2:
        return state

    ai_responses = [m for m in state["messages"] if m.type == "ai"]
    user_messages = [m for m in state["messages"] if m.type == "human"]

    if not ai_responses or not user_messages:
        return state

    last_user_message = user_messages[-1].content
    last_ai_response = ai_responses[-1].content

    extract_prompt = f"""Look at this conversation and extract any facts worth remembering about the user.

    User: {last_user_message}
    Assistant: {last_ai_response}

    List each fact on a new line starting with a dash (-).
    Only include clear, factual information about the USER (not about the assistant).
    If there are no facts to remember, respond with: NONE

    Examples of good facts:
    - User's name is Arjun
    - User works as a software engineer
    - User enjoys coding
    - User is learning AI

    Examples of bad facts (don't include these):
    - The assistant was helpful
    - We had a conversation
    - The user asked a question"""

    result = llm.with_structured_output(UserMemory).invoke(extract_prompt)

    stored_count = sum(
        store_manager.save(user_id, fact) for fact in result.facts if fact
    )

    print(f"Stored {stored_count} new facts for user {user_id}")

    return state

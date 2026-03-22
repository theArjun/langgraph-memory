from langchain_core.messages import HumanMessage

from ai.graph import graph

THREAD_ID = "chat_001"
USER_ID = "arjun"


config = {"configurable": {"thread_id": THREAD_ID, "user_id": USER_ID}}


if __name__ == "__main__":
    user_query_1 = "My name is Arjun Adhikari and I work as software engineer."
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_query_1)]},
        config=config,
    )

    ai_response = result["messages"][-1].content
    print(ai_response)

    user_query_2 = "What do you know about me?"
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_query_2)]},
        config=config,
    )
    ai_response = result["messages"][-1].content

    user_query_3 = "What could be my hobbies ?"
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_query_3)]},
        config=config,
    )
    ai_response = result["messages"][-1].content

    print(ai_response)

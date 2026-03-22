from ai.graph import graph_manager

THREAD_ID = "chat_001"
USER_ID = "arjun"

if __name__ == "__main__":
    response = graph_manager.invoke(
        USER_ID, THREAD_ID, "My name is Arjun Adhikari and I work as software engineer."
    )
    print(response)

    response = graph_manager.invoke(USER_ID, THREAD_ID, "What do you know about me?")
    print(response)

    response = graph_manager.invoke(USER_ID, THREAD_ID, "What could be my hobbies?")
    print(response)

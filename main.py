from ai.graph import graph_manager

THREAD_ID = "chat_001"
USER_ID = "arjun"

if __name__ == "__main__":
    print("💬 Chat started. Type 'exit' or 'quit' to stop.\n")
    while True:
        message = input("🧑 You: ").strip()
        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            break
        response = graph_manager.invoke(USER_ID, THREAD_ID, message)
        print(f"\n🤖 Assistant: {response}\n")

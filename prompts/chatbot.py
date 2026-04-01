import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError(
        "COHERE_API_KEY is missing. Add it to your .env file before running this script."
    )

primary_model_name = os.getenv("COHERE_MODEL", "command-a-03-2025")
fallback_model_name = os.getenv("COHERE_FALLBACK_MODEL", "command-r7b-12-2024")
current_model_name = primary_model_name


def build_model(model_name: str) -> ChatCohere:
    return ChatCohere(
        model=model_name,
        temperature=0.3,
        cohere_api_key=cohere_api_key,
    )


model = build_model(current_model_name)

chat_history = [SystemMessage(content="You are a helpful AI assistant")]

print("Type 'exit' to quit.")
print(f"Using Cohere model: {current_model_name}")

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nEnding chat.")
        break

    if not user_input:
        continue

    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    try:
        result = model.invoke(chat_history)
    except Exception as err:
        error_text = str(err)
        should_fallback = (
            current_model_name != fallback_model_name
            and (
                "status_code: 404" in error_text
                or "was removed" in error_text
                or "not found" in error_text.lower()
            )
        )

        if should_fallback:
            print(
                f"Model '{current_model_name}' is unavailable. "
                f"Switching to '{fallback_model_name}'."
            )
            current_model_name = fallback_model_name
            model = build_model(current_model_name)
            result = model.invoke(chat_history)
        else:
            raise

    response_text = result.content if isinstance(result.content, str) else str(result.content)

    chat_history.append(AIMessage(content=response_text))
    print("AI:", response_text)

print("\nConversation ended.")
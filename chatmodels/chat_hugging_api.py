from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatCohere(
    model="command-a-03-2025",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

res = model.invoke("What is the capital of India?")
print(res.content)
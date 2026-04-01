from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
embedding = CohereEmbeddings(model="embed-english-v3.0",
cohere_api_key=os.getenv("COHERE_API_KEY"))
documents=[
    "delhi is the capital of india",
    "kolkata is the cleanest city of india",
    "france is the capital of paris"
]
# text =  "Delhi is the capital of india"

vector = embedding.embed_documents(documents)

print(vector)


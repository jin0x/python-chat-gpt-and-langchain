import os
import pinecone
from langchain.vectorstores import Pinecone
from app.chat.embeddings.openai import embeddings

pinecone.init(
    api_key=os.get_env("PINECONE_API_KEY"),
    environment=os.get_env("PINECONE_ENV_NAME"),
)

vector_store = Pinecone.from_existing_index(
    os.get_env("PINECONE_INDEX_NAME"), embeddings
)
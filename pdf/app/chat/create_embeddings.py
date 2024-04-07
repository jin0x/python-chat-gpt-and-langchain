import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from app.chat.vector_stores.pinecone import vector_store

def create_embeddings_for_pdf(pdf_id: str, pdf_path: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split(text_splitter)

    print(os.getenv("PINECONE_API_KEY"))
    print(os.getenv("PINECONE_ENV_NAME"))
    print(os.getenv("PINECONE_INDEX_NAME"))
    print(os.getenv("OPENAI_API_KEY"))

    # vector_store.add_documents(docs)

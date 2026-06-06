from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

from config import MISTRAL_API_KEY, VECTORSTORE_PATH

def get_embedder():
    return MistralAIEmbeddings(
        model="mistral-embed",   # available embedding model
        api_key=MISTRAL_API_KEY
    )

def build_vectorstore(chunks: list) -> FAISS:
    embedder = get_embedder()
    print("Generating embeddings... (this may take a minute)")
    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"FAISS index saved to {VECTORSTORE_PATH}")
    return vectorstore

def load_vectorstore() -> FAISS:
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError("No FAISS index found. Run main.py first.")
    embedder = get_embedder()
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embedder,
        allow_dangerous_deserialization=True
    )


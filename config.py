import os
from dotenv import load_dotenv

load_dotenv()


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

EMBEDDING_MODEL = "mistral-embed"   # use Mistral’s embedding model

# 💬 LLM model (Mistral)
LLM_MODEL = "mistral-tiny"          # or "mistral-medium", "mistral-large" depending on your plan


VECTORSTORE_PATH = "vectorstore/faiss_index"
DATA_DIR = "data/"

# 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5

# 
SYSTEM_PROMPT = """You are a helpful assistant that answers questions 
based ONLY on the provided context from the documents.
If the answer is not in the context, say 'I could not find this 
in the provided documents.' Do not hallucinate.

Context:
{context}

Question: {question}
Answer:"""


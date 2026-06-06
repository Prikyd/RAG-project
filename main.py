from utils.pdf_loader import load_pdfs, chunk_documents
from utils.embedder import build_vectorstore

if __name__ == "__main__":
    print("=== RAG Document Indexer ===")
    documents = load_pdfs()
    chunks = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks)
    print("Indexing complete! Run: streamlit run app.py")
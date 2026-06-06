from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR
import os

def load_pdfs(data_dir: str = DATA_DIR) -> list:
    """Load all PDFs from the data directory."""
    documents = []
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")
    
    for pdf_file in pdf_files:
        path = os.path.join(data_dir, pdf_file)
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)
        print(f"Loaded: {pdf_file} ({len(docs)} pages)")
    
    return documents

def chunk_documents(documents: list) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
             length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks
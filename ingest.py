"""Load PDFs from data/, chunk them, store in Chroma, and save chunks for BM25."""
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
CHUNKS_PATH = Path("chunks.pkl")


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest():
    docs = []
    for pdf in DATA_DIR.glob("*.pdf"):
        docs.extend(PyPDFLoader(str(pdf)).load())
    if not docs:
        raise SystemExit("No PDFs found in data/. Add some and rerun.")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800, chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        chunks, get_embeddings(), persist_directory=CHROMA_DIR
    )
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)  # reused to build the BM25 index
    print(f"Ingested {len(docs)} pages -> {len(chunks)} chunks")


if __name__ == "__main__":
    ingest()

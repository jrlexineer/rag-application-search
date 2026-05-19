import os
from pathlib import Path
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Load API keys from .env into environment variables
load_dotenv()

# Connect to OpenAI (uses OPENAI_API_KEY from .env automatically)
client = OpenAI()

# Set up persistent ChromaDB storage on disk
chroma_client = chromadb.PersistentClient(path="./data/processed/chroma")
collection = chroma_client.get_or_create_collection(
    "applications",
    metadata={"hnsw:space": "cosine"}
)


def chunk_text(text, max_tokens=500):
    """Split text into roughly token-sized chunks on paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for p in paragraphs:
        # Rough char-to-token ratio of 4:1
        if len(current) + len(p) < max_tokens * 4:
            current += p + "\n\n"
        else:
            chunks.append(current.strip())
            current = p + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks


def embed_and_store(filepath):
    """Read a file, chunk it, embed each chunk, store in ChromaDB."""
    text = Path(filepath).read_text(encoding="utf-8")
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue  # skip empty chunks
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding
        collection.add(
            ids=[f"{filepath}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"filename": str(filepath), "chunk_index": i}]
        )
        print(f"  Stored chunk {i} ({len(chunk)} chars)")


if __name__ == "__main__":
    for filepath in Path("data/raw").glob("**/*.txt"):
        print(f"Ingesting {filepath}")
        embed_and_store(filepath)
    print(f"\nDone. Collection now has {collection.count()} chunks.")
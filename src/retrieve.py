import sys
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./data/processed/chroma")
collection = chroma_client.get_or_create_collection("applications")


def embed_query(query):
    """Embed a query string using the same model as ingestion."""
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding


def retrieve(query, k=3):
    """Return the top-k most similar chunks to the query."""
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    # Unpack the nested structure ChromaDB returns
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


if __name__ == "__main__":
    # Take query from command line, or use a default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "experience with data analysis"

    print(f"Query: {query}\n")
    print(f"Top results:\n" + "=" * 60)

    hits = retrieve(query, k=3)
    for i, hit in enumerate(hits, 1):
        filename = hit["metadata"]["filename"]
        distance = hit["distance"]
        print(f"\n[{i}] {filename} (distance: {distance:.4f})")
        print("-" * 60)
        print(hit["document"][:400] + ("..." if len(hit["document"]) > 400 else ""))
        print()
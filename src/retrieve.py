import sys
import re
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./data/processed/chroma")
collection = chroma_client.get_or_create_collection(
    "applications",
    metadata={"hnsw:space": "cosine"}
)


def embed_query(query):
    """Embed a query string using the same model as ingestion."""
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding


def retrieve(query, k=3):
    """Return the top-k most similar chunks to the query (vector search)."""
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


# --- BM25 keyword search over the same chunks stored in Chroma ---
_bm25 = None
_bm25_corpus = None  # aligned list of {id, document, metadata}


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _build_bm25():
    """Pull every chunk out of Chroma once and build a BM25 index over them."""
    global _bm25, _bm25_corpus
    data = collection.get(include=["documents", "metadatas"])
    _bm25_corpus = [
        {"id": data["ids"][i],
         "document": data["documents"][i],
         "metadata": data["metadatas"][i]}
        for i in range(len(data["ids"]))
    ]
    _bm25 = BM25Okapi([_tokenize(c["document"]) for c in _bm25_corpus])


def bm25_search(query, k=3):
    """Top-k chunks by BM25 keyword score. Same dict shape as retrieve()."""
    if _bm25 is None:
        _build_bm25()
    scores = _bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{
        "id": _bm25_corpus[i]["id"],
        "document": _bm25_corpus[i]["document"],
        "metadata": _bm25_corpus[i]["metadata"],
        "distance": float(scores[i]),  # BM25 score, not a cosine distance
    } for i in ranked]


def hybrid_retrieve(query, k=3, rrf_k=60):
    """Chunk-level RRF fusion of vector + BM25. Drop-in replacement for retrieve()."""
    if _bm25 is None:
        _build_bm25()
    n = len(_bm25_corpus)

    vres = collection.query(query_embeddings=[embed_query(query)], n_results=n)
    vector_ids = vres["ids"][0]

    scores = _bm25.get_scores(_tokenize(query))
    bm25_ids = [_bm25_corpus[i]["id"]
                for i in sorted(range(n), key=lambda i: scores[i], reverse=True)]

    fused = {}
    for ranking in (vector_ids, bm25_ids):
        for rank, cid in enumerate(ranking, 1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    top = sorted(fused, key=fused.get, reverse=True)[:k]
    by_id = {c["id"]: c for c in _bm25_corpus}
    return [{
        "id": cid,
        "document": by_id[cid]["document"],
        "metadata": by_id[cid]["metadata"],
        "distance": fused[cid],  # fused RRF score (higher = better); score.py ignores it
    } for cid in top]


if __name__ == "__main__":
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

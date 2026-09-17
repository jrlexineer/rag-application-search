# rag-application-search

A retrieval-augmented search system over my own job application materials — resumes, cover letters, and supplemental answers. About 100 lines of Python, no frameworks. Built to practice the part of RAG that actually determines quality: measuring retrieval performance and diagnosing failures per-query.

It proved useful almost immediately: when I asked which application talked about Gorgias, it noticed that the file named `cover-letter-GORGIAS.txt` was actually addressed to Vouch. That's a catch I wouldn't have gotten from grep.

## Results

| Configuration | MRR | Recall@1 | Recall@3 |
|---|---|---|---|
| Baseline (dense only) | 0.569 | 0.259 | 0.630 |
| Hybrid (dense + BM25, RRF) | 0.637 | — | — |

Evaluated on 9 hand-labeled questions over a 7-document corpus, with ground truth judged by reading every chunk by hand.

Per-query failure analysis on the baseline showed misses concentrated in one pattern: concept queries without distinctive keyword anchors. That's the failure mode lexical retrieval addresses, so I added BM25 alongside dense retrieval, fused with reciprocal rank fusion. The metric moved because the diagnosis was right, not because I tuned blindly.

## Why this corpus

Most RAG tutorials use Paul Graham essays or Wikipedia dumps. I wanted something I actually needed to query — questions like "which cover letters mention dbt" or "what eval framings have I used across applications" had real stakes for me. The corpus started at three documents and grew to seven by evaluation time. Small on purpose: small enough to read every retrieved result by hand and judge whether the system was actually working.

## How it works

Three scripts, in order. No LangChain, no LlamaIndex.

- **`ingest.py`** reads `.txt` files from `data/raw/`, splits them into ~500-token chunks on paragraph boundaries, embeds each chunk with OpenAI's `text-embedding-3-small`, and stores the result in a local ChromaDB collection (cosine distance).
- **`retrieve.py`** embeds the query and runs hybrid retrieval: dense nearest-neighbor search against ChromaDB alongside lexical scoring with BM25, fused with reciprocal rank fusion (RRF) into a single top-k result set, returned with filenames and scores.
- **`generate.py`** formats the top chunks into a context block and asks Claude to answer the query using only those chunks, with citations to the original filenames.

## What I learned building it

**The distance metric mattered more than the chunker.** ChromaDB defaults to squared L2 distance, which measures magnitude as well as direction — wrong for normalized text embeddings. My first queries returned scores in a 1.4–1.8 band with mostly arbitrary ranking. Switching the collection to cosine (`metadata={"hnsw:space": "cosine"}`) moved scores into an interpretable 0.2–0.8 range and made top-1 usually correct. Single biggest quality jump in the build.

**Retrieval needs to be good enough, not perfect.** The Gorgias-vs-Vouch catch happened on a query whose best chunk scored a mediocre 0.758 distance. A pure-retrieval system returning chunks directly would have surfaced nothing — the synthesis layer is where the observation emerged. This changed how I think about RAG quality: retrieval's job is to get the relevant material into the context window, not to rank perfectly.

**Filename metadata is half the value.** Adding `{"filename", "chunk_index"}` to the schema was a five-character change that turned "found a relevant excerpt" into "found a relevant excerpt in this specific file." That's the difference between a search demo and a useful tool.

**The naive chunker was fine.** Split on `\n\n`, batch to ~500 tokens, ship. I'd been told chunking is the hard part of RAG; measurement showed it wasn't the bottleneck on this corpus. The eval harness is what makes that a finding instead of a guess.

## What's broken or limited

- **Small corpus.** Seven documents. The system works partly because the model synthesizes well from sparse context, not because retrieval is sharp at scale.
- **No conversation memory.** Every query is fresh; follow-ups are impossible.
- **Naive chunker.** No real token counting, no sentence-boundary awareness. Single bullets become tiny chunks; dense paragraphs overshoot.
- **Manual ingestion.** Input requires clean `.txt`. A real version would handle `.docx` and `.pdf`.

## What's next

Expand the corpus to 15–20 applications and re-run the eval to see if retrieval quality holds at scale; rewrite the chunker with `tiktoken` and document-structure awareness; add a chat loop so follow-ups work. The test of whether this was worth building stays the same: whether it becomes the primary interface I use when applying for things.

## Setup

```
git clone https://github.com/jrlexineer/rag-application-search.git
cd rag-application-search
python -m venv venv
.\venv\Scripts\Activate.ps1   # or: source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Put `.txt` files in `data/raw/`, then:

```
python src/ingest.py
python src/generate.py "your question here"
```

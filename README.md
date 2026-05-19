# rag-application-search

A retrieval-augmented search system over my own job application materials — resumes, cover letters, and supplemental answers from the last few weeks. I built it in about an hour \[DATE/CONTEXT — "as a forcing function to learn RAG" / "on a Sunday afternoon" / etc.], and it immediately did something useful: when I asked it which application talked about Gorgias, it noticed that the file named cover-letter-GORGIAS.txt was actually addressed to Vouch. That's the kind of self-audit I couldn't have gotten from grep.

## Why this corpus

Most RAG tutorials use Paul Graham essays or Wikipedia dumps. I wanted to use something I actually cared about querying. Job application materials were the obvious choice: they were already organized on my machine, the licensing was trivially clean, and I genuinely wanted to be able to ask things like "what eval framings have I used across applications" or "which cover letters mention dbt" without having to re-read everything.

The corpus is small on purpose. Three documents, six chunks, two embedded resumes and one cover letter. Enough to test the pipeline; small enough that I could read every retrieved result by hand and judge whether the system was actually working.

##How it works

Three scripts, in order:



ingest.py reads .txt files from data/raw/, splits them into \~500-token chunks on paragraph boundaries, embeds each chunk with OpenAI's text-embedding-3-small, and stores the result in a local ChromaDB collection.

retrieve.py takes a query string, embeds it with the same model, and returns the top-k nearest chunks from the collection along with their filenames and distance scores.

generate.py calls retrieve.py, formats the top chunks into a context block, and asks Claude Sonnet 4.6 to answer the query using only those chunks — with citations to the original filenames.



No frameworks. No LangChain, no LlamaIndex. The whole thing is about 100 lines of Python and three API calls.

##What I learned building it

A few things I didn't expect.

ChromaDB's default distance metric is wrong for text. My first round of queries returned distance scores in the 1.4–1.8 range across the board, and the ranking felt mostly arbitrary. The issue turned out to be that ChromaDB defaults to squared L2 distance, which measures magnitude as well as direction. For OpenAI's normalized text embeddings, you want cosine distance — which measures only the angle between vectors. After switching to cosine (metadata={"hnsw:space": "cosine"} at collection creation), distances landed in a much more interpretable 0.2–0.8 range and the top-1 result was usually the right one. This was the single biggest quality jump in the whole build.

Retrieval is the weak link; the LLM rescues a lot. Several of my queries returned chunks with distance scores around 0.75 — i.e. "somewhat related, not a great match." But the generated answers were still excellent, because Claude was doing real synthesis across imperfect retrieval. The Gorgias-vs-Vouch catch happened on a query where the top retrieved chunk had a 0.758 distance. If I'd built a pure-retrieval system that returned chunks directly, I would have missed that observation entirely. This shifted how I think about RAG quality: retrieval needs to be good enough to get the relevant material in the context window, not good enough to rank perfectly.

Filename metadata is half the value. I almost didn't include it. The first version of the schema just stored the chunk text and an ID. Adding {"filename": ..., "chunk\_index": ...} as metadata was a five-character change that turned "the system found a relevant excerpt" into "the system found a relevant excerpt from cover-letter-GORGIAS.txt." That's the difference between a search engine and a useful tool for someone managing parallel applications.

The naive chunker is fine. I spent zero effort on a sophisticated chunker. Split on \\n\\n, batch paragraphs until \~500 tokens, ship it. There are obvious problems with this — bullet points become tiny chunks, dense paragraphs don't get split — but for a corpus this small, the chunker wasn't the bottleneck. I'd been told to expect chunking to be the hardest part of RAG; in practice, distance metric selection mattered more.

##What's broken or limited

A short and honest list:



Corpus is tiny. Three documents, six chunks. The system works because Claude is good at synthesizing from sparse context, not because retrieval is sharp.

No hybrid search. Pure semantic retrieval misses exact keyword matches. A query for a specific company name might not surface the file that mentions it once.

No conversation memory. Every python src/generate.py is a fresh query. Follow-ups are impossible.

Naive chunker. No actual token counting, no sentence-boundary respect. Single bullet points become their own chunks; long paragraphs blow past the target size.

Manual .txt conversion. I had to copy-paste from Word docs into Notepad to get clean input. A real version would handle .docx and .pdf.



##What's next

In rough order: expand the corpus to 15–20 applications and see if retrieval quality genuinely improves; rewrite the chunker to use tiktoken and respect document structure; add a chat loop so follow-up questions work; add hybrid retrieval (BM25 alongside embeddings) for exact keyword queries. If those go well, the system becomes the actual primary interface I use when applying for things — which is the test of whether this was worth building.

##Setup

```
git clone https://github.com/jrlexineer/rag-application-search.git

cd rag-application-search

python -m venv venv

.\\venv\\Scripts\\Activate.ps1  # or source venv/bin/activate

pip install -r requirements.txt
```
Create a .env file with:
```
OPENAI\_API\_KEY=sk-...

ANTHROPIC\_API\_KEY=sk-ant-...
```
Put .txt files in data/raw/, then:
```
python src/ingest.py

python src/generate.py "your question here"
```

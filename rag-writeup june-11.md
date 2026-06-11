# Retrieval over a corpus I know cold: a measured walk from a dense baseline to hybrid + contextual embeddings

Most retrieval writeups report a single configuration and a single number. This one reports the path: a dense baseline, two changes applied one at a time, a negative result I almost talked myself out of recording, and one query that refused to improve no matter what I did to it. The corpus is small and personal on purpose — 7 of my own job-application documents (resumes, cover letters, supplemental answers) — because I know every document well enough to hand-grade every retrieval and write an answer key I trust. The goal was never a product. It was to be able to say, with evidence, *which* change did the work, and to be wrong out loud when my predictions missed.

The headline: starting from a vector-only baseline, adding a lexical channel and then document-level context to each chunk moved mean reciprocal rank from **0.569 to 0.796** and roughly doubled recall@1, from **0.259 to 0.481**. But the aggregate is the least interesting part. The per-question movement, and the one question that didn't move, are where the actual lessons are.

## System

The whole system is about 100 lines of Python with no orchestration framework — deliberately, so that nothing is hidden behind a library default I didn't choose. Three scripts:

- `ingest.py` reads each document, splits it into chunks, embeds each chunk, and writes `(id, embedding, document, metadata)` rows into a persistent ChromaDB collection.
- `retrieve.py` embeds a query and runs a cosine-similarity lookup, unpacking ChromaDB's column-oriented response into a flat list of hits with their distances.
- `generate.py` formats the top chunks into a context block and hands them to Claude with a system prompt that instructs it to answer only from the provided excerpts, cite the source filename for every fact, and say so explicitly when the excerpts don't contain the answer rather than guessing.

Embeddings are OpenAI `text-embedding-3-small` (1536 dimensions). The vector store is ChromaDB configured for cosine space, which it reports as a distance: `distance = 1 − cosine_similarity`, so lower is more similar. The generation model is Claude, but generation is downstream of everything that matters here and I never measure it — more on that below.

**Chunking.** Documents are split on paragraph boundaries and greedily packed into chunks of up to roughly 500 tokens, estimated with a 4-character-per-token heuristic. Paragraph-boundary splitting matters for this corpus: a cover letter's argument lives in its paragraphs, and cutting mid-paragraph to hit an exact token count would strand the second half of a claim in a chunk that no longer says who it's about. Seven documents produce on the order of twenty chunks. That's small, and the small N shapes how I read everything that follows.

## The evaluation harness

I measure retrieval, not generation. If the correct chunk never enters the candidate set, no amount of model quality recovers the answer — so the only thing I can independently break or fix here is which chunks come back and in what order. Measuring generation on top would entangle two failure sources and tell me less about either.

The harness is 9 hand-written questions paired with an answer key naming the correct document(s) for each. For every query I record the **rank of the first correct result**, then compute two metrics:

- **MRR (mean reciprocal rank)** — the mean over queries of `1 / rank_of_first_correct`. A correct answer at rank 1 contributes 1.0; rank 2 contributes 0.5; rank 5 contributes 0.2. MRR answers a ranking-quality question: when retrieval finds the right thing, how high does it put it? It punishes anything below rank 1 sharply and non-linearly, which is the right shape — the gap between rank 1 and rank 2 should hurt more than the gap between rank 4 and rank 5.
- **Recall@k** — the fraction of correct items appearing in the top k. Recall@1 is "did the single best hit count"; recall@3 is "was it in the top three." This is a coverage question: did retrieval find the thing at all? Several of my questions are deliberately multi-answer ("which applications reference Salesforce architecture"), so recall@k averages over all correct items, not just the first — which is why recall@1 can sit below what a table of first-hit ranks alone would suggest.

You need both. MRR with high recall means a ranking problem; low recall means a coverage problem; and the prescription for each is different.

The questions span the two regimes I care about: lexically anchored queries that name a company, role, or distinctive term, and abstract conceptual queries that describe a quality ("which resume goes deepest on the technical stack") without naming anything that appears verbatim in the target. One question is an intentional trap, described in the failure analysis.

## Baseline

Vector-only, top-k cosine retrieval:

```
MRR        0.569
recall@1   0.259
recall@3   0.630
```

Read honestly, recall@1 of 0.259 means the correct chunk is the single best hit only about a quarter of the time. Three queries out of four, my top result is not my most relevant chunk. That's not good — but the jump to recall@3 of 0.630 is the signal. The right chunk is usually *in* the result set, just not ranked first. So this is at least as much a ranking problem as a coverage problem, and MRR of 0.569 agrees: when retrieval lands the right chunk, it tends to land it near the top rather than buried at rank 4 or 5. A coverage problem would have looked like low recall@3 too. This doesn't. That diagnosis — ranking, not coverage — is what pointed at the interventions worth trying.

## Failure analysis

The misses are not random, and the pattern is the entire justification for what I changed next.

**Anchored queries retrieved well.** "Which application was for Airtable" lands at rank 1: the token *Airtable* sits in the document body, dominates the query embedding, and pins cleanly to the one chunk that shares it. Anchored queries have a rare token doing most of the semantic work, and rare tokens are exactly what an embedding can localize.

**Abstract queries retrieved worse.** "Which resume goes deepest on the technical stack" has no rare anchor. Its similarity smears across a half-dozen documents that all gesture at competence, and nothing separates. The embedding is being asked to rank documents on a quality none of them states in so many words, against neighbors that are genuinely close in meaning.

Then there's the trap, and it's worth stating precisely because it's easy to get backwards. One cover letter is filed as `cover-letter-GORGIAS.txt`, but the letter is addressed to **Vouch**. The string "Gorgias" appears *only in the filename* — which is never embedded. So my trick question, "which company was the cover letter actually written for," has no content signal pointing anywhere useful; the body says Vouch. This is the most useful single fact the project surfaced: **embeddings are linguistically literal — the vector reflects only the characters you feed the embedder, not the filename, not the metadata, not your intent.** I'll return to this query twice more, because it's the one that resists every fix I throw at it.

## Interventions

Two changes, each measured in isolation against the same nine questions, plus one I expected to help and didn't.

**BM25 hybrid (lexical + dense, fused with RRF).** BM25 is a probabilistic keyword ranker — weighted TF-IDF with two knobs, `k1` (saturates the contribution of repeated terms) and `b` (normalizes for document length). It matches by literal, rarity-weighted word overlap; the embedding matches by meaning. They fail in *different* ways, which is the whole argument for running both. I fuse their rankings with Reciprocal Rank Fusion: each document's score is the sum over both lists of `1 / (rrf_k + rank)`, with `rrf_k = 60`. RRF is rank-based, not score-based, so it doesn't require the two channels to be on a comparable scale — it only trusts their orderings. The win I expected was protection for the anchored queries plus recovery of exact terms the embedding glosses over.

**Contextual embeddings.** Following Anthropic's contextual-retrieval recipe: before embedding each chunk, I prepend a short (~50–100 token) LLM-generated blurb situating it — which file, which role, which company, where in the document it sits. The blurb is stored *as part of the document text*, not just used for embedding, so the BM25 channel reads it back out too; both channels get the document-level identity for free. The cost is one Claude call per chunk at ingest time — roughly twenty calls for this corpus, pennies total, paid once. The intent was to give abstract queries more semantic surface to match against: a chunk that says "I shipped X" becomes a chunk that says "from Josh's resume tailored to an Applied AI Engineer role, describing shipping X."

**A negative result I'm keeping.** I also tried collapsing retrieval to the single best chunk per document before fusing — "document-level" RRF — on the theory that near-duplicate chunks from the same file were crowding each other out of the top ranks. It backfired. With `rrf_k = 60` over only 7 documents, every document's fused score lands in the razor-thin band between `1/61` and `1/67`; at that compression RRF barely discriminates, collapsing toward "did you appear in both lists at all" and throwing away rank order. Noise documents floated up and pushed correct ones out of the top 3. Recall@3 fell from 0.685 to 0.519. The hypothesis was wrong; I reverted it. If I ever revisit document-level fusion I'd have to drop `rrf_k` to something like 10 to restore discrimination at this corpus size — but at 7 documents it isn't worth it. I'm reporting it because the harness existing is precisely what let me catch it instead of shipping it.

## Results

Each stage measured against the same nine questions:

```
                  vector    hybrid    hybrid + contextual
MRR                0.569     0.637     0.796
recall@1           0.259     0.315     0.481
recall@3           0.630     0.685     0.907
```

Per-question rank of first correct, for the hybrid stage (the run I instrumented most closely):

| Query                                                        | Rank |
|--------------------------------------------------------------|:----:|
| mentions Claude API workflow design                          |  1   |
| was for an Applied AI Engineer role                          |  5   |
| was for Airtable                                             |  1   |
| mention building an eval harness                             |  1   |
| reference Salesforce architecture                            |  2   |
| were for CSM roles                                           |  1   |
| emphasizes customer-facing deployment / enterprise delivery  |  2   |
| goes deepest on the technical stack                          |  5   |
| which company the cover letter was actually written for      |  3   |

Hybrid beat the baseline on every aggregate metric, but the per-question view is where it earns or loses trust. The clear win is **Salesforce architecture**, which climbed from the bottom of the list to rank 2: fusion demoted resume chunks that were matching on a stray keyword and lifted the cover letters that actually answered the question. That is the noise-demotion mechanism behaving exactly as predicted. The clear cost is **Applied AI Engineer**, which *regressed* — a query winning at rank 1–2 under pure vector dropped to rank 5. At rank 5 it contributes 0.2 to MRR instead of 1.0; if it had held at rank 1, hybrid MRR would have been around 0.73 instead of 0.637. One change produced both the best and the most expensive movement on the board, and I'd have seen neither if I were only watching the aggregate.

Then contextual embeddings did the heavy lifting. MRR rose from 0.637 to 0.796; recall@3 reached 0.907; recall@1 nearly doubled off the original baseline, so the system now puts a correct document first about half the time rather than a third. Most queries that had been landing at rank 3 or 5 moved up to rank 1 or 2. Decomposed: the lexical channel bought a modest, defensible bump and a more robust failure profile; the document-level context is what actually moved the project.

## The query that didn't move

The most informative result is the one that stayed put. I had framed contextual labels as the fix for abstract queries, and I specifically predicted that the Vouch trap would finally jump to rank 1 — its blurb now literally names Vouch, so the company is in the embedded text at last. It didn't move.

The reason is the lesson. The question — "which company was the cover letter written for" — doesn't name a company; it asks about one in the abstract. And once *every* cover letter carries a blurb announcing the company it was written for, all of them look equally like the answer to a question about which company a cover letter was written for. The labels help any query that asks about a *distinguishing* detail: a named company, a specific role, a particular skill. They actively hurt a query about a *shared category*, because now every member of that category introduces itself the same way. Contextual labels made my cover letters easier to find as a group and no easier to tell apart. That tradeoff is invisible in the aggregate — MRR went up overall — and only shows up if you track the individual query you were sure you'd fixed.

## Limitations and threats to validity

This is nine questions over seven documents. The movements are large enough to trust the direction, not the third decimal: at this N a single rank flip is worth roughly 0.09 of MRR, so I report a trend, not a benchmark, and I wouldn't defend "0.796" past its first digit. The same person wrote the documents, the questions, and the answer key, which risks a query phrasing that unconsciously favors what I know retrieves well; a cleaner design would have someone else write the evaluation set. I measure retrieval only — end-to-end answer quality could move differently, since a strong model sometimes recovers from a mediocre context and a weak one squanders a perfect one. And every number is corpus-specific: the `rrf_k` mushiness that sank document-level fusion is a small-corpus artifact, and the contextual-label-blurring effect depends on how categorically similar your documents are. On a corpus of genuinely distinct documents, the labels might have no downside at all. Nothing regressed on net between stages, so the reported gains didn't come at a hidden cost I failed to look for — but "I looked and didn't find one" is a weaker claim at N=9 than it would be at N=900.

## What I'd keep

If I deleted everything but one habit, it would be the harness. Without the test questions, "it feels sharper now" would have been the entire conclusion — and it would have been right, and useless. The harness is what let me see that of two changes I was equally excited about, one did most of the work; that the change I was most confident in was the one that failed on the hardest query; and that a sensible-sounding optimization quietly made retrieval worse. None of those are facts about retrieval. They're facts about measurement, and being able to be precisely, publicly wrong is what made any of the retrieval lessons legible in the first place. The techniques are replaceable. The discipline of changing one thing and watching the same numbers move is the part worth carrying to the next system.

# eval/score.py

import sys
sys.path.append("src")
from retrieve import retrieve

ANSWER_KEY = [
    {
        "query": "which company was the cover letter actually written for?",
        "correct": "data\\raw\\cover-letter-GORGIAS.txt",
    },
    {
        "query": "which resume emphasizes customer-facing deployment and enterprise delivery?",
        "correct": "data\\raw\\cognition-resume.txt",
    },
    {
        "query": "which resume goes deepest on the technical stack and building solutions?",
        "correct": "data\\raw\\OWNER-resume.txt",
    },
]

def filenames_from_hits(hits):
    return [hit["metadata"]["filename"] for hit in hits]

def score():
    reciprocal_ranks = []
    for item in ANSWER_KEY:
        hits = retrieve(item["query"], k=3)
        returned = filenames_from_hits(hits)

        if item["correct"] in returned:
            rank = returned.index(item["correct"]) + 1
        else:
            rank = None

        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        print(f"\nQ: {item['query']}")
        print(f"   correct file: {item['correct']}")
        print(f"   got back:     {returned}")
        print(f"   correct file's rank: {rank if rank else 'NOT FOUND'}")

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"\n{'='*40}")
    print(f"MRR (baseline): {mrr:.3f}")

if __name__ == "__main__":
    score()

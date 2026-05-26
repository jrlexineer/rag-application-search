# eval/score.py

import sys
sys.path.append("src")          # lets us import your retrieve.py from the src folder
from retrieve import retrieve

# Your answer key. The filenames must match EXACTLY what the system stores —
# note the data\raw\ prefix, because that's what your metadata uses.
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
    """Pull just the filename string out of each retrieved hit, in rank order."""
    return [hit["metadata"]["filename"] for hit in hits]

def score():
    for item in ANSWER_KEY:
        hits = retrieve(item["query"], k=3)
        returned = filenames_from_hits(hits)

        # Where did the correct file land in the ranked list? (1 = top result)
        if item["correct"] in returned:
            rank = returned.index(item["correct"]) + 1
        else:
            rank = None   # correct file didn't come back at all

        print(f"\nQ: {item['query']}")
        print(f"   correct file: {item['correct']}")
        print(f"   got back:     {returned}")
        print(f"   correct file's rank: {rank if rank else 'NOT FOUND'}")

if __name__ == "__main__":
    score()

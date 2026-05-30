import sys
sys.path.append("src")
from retrieve import hybrid_retrieve as retrieve

ANSWER_KEY = [
    {"query": "which application explicitly mentions Claude API workflow design",
     "correct": ["data\\raw\\cover-letter-amplitude-csm.txt"]},
    {"query": "which application was for an Applied AI Engineer role",
     "correct": ["data\\raw\\resume-paraform-aae.txt"]},
    {"query": "which application was for Airtable",
     "correct": ["data\\raw\\cover-letter-airtable-AI-analytics.txt"]},
    {"query": "which applications mention building an eval harness",
     "correct": ["data\\raw\\cover-letter-airtable-AI-analytics.txt",
                 "data\\raw\\cover-letter-coalition-csm.txt",
                 "data\\raw\\resume-paraform-aae.txt"]},
    {"query": "which applications reference Salesforce architecture",
     "correct": ["data\\raw\\cover-letter-airtable-AI-analytics.txt",
                 "data\\raw\\cover-letter-coalition-csm.txt"]},
    {"query": "which applications were for CSM roles",
     "correct": ["data\\raw\\cover-letter-coalition-csm.txt",
                 "data\\raw\\cover-letter-amplitude-csm.txt"]},
    {"query": "which resume emphasizes customer-facing deployment and enterprise delivery",
     "correct": ["data\\raw\\cognition-resume.txt"]},
    {"query": "which resume goes deepest on the technical stack and building solutions",
     "correct": ["data\\raw\\OWNER-resume.txt", "data\\raw\\resume-paraform-aae.txt"]},
    {"query": "which company was the cover letter actually written for",
     "correct": ["data\\raw\\cover-letter-GORGIAS.txt"]},
]

KS = (1, 3, 5)

def files(hits):
    return [h["metadata"]["filename"] for h in hits]

def recall_at_k(returned, correct, k):
    topk = returned[:k]
    hits = sum(1 for c in correct if c in topk)
    return hits / len(correct)

def first_rank(returned, correct):
    for i, f in enumerate(returned, 1):
        if f in correct:
            return i
    return None

def score():
    rrs, recalls = [], {k: [] for k in KS}
    for item in ANSWER_KEY:
        returned = files(retrieve(item["query"], k=max(KS)))
        correct = item["correct"]
        rank = first_rank(returned, correct)
        rrs.append(1.0 / rank if rank else 0.0)
        for k in KS:
            recalls[k].append(recall_at_k(returned, correct, k))
        print(f"\nQ: {item['query']}")
        print(f"   rank of first correct: {rank if rank else 'NOT FOUND'}")
    mrr = sum(rrs) / len(rrs)
    print(f"\n{'='*40}")
    print(f"MRR: {mrr:.3f}")
    for k in KS:
        print(f"recall@{k}: {sum(recalls[k])/len(recalls[k]):.3f}")

if __name__ == "__main__":
    score()

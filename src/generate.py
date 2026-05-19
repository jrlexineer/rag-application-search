import sys
from anthropic import Anthropic
from dotenv import load_dotenv
from retrieve import retrieve

load_dotenv()

anthropic_client = Anthropic()

SYSTEM_PROMPT = """You are an assistant that answers questions about Josh's job application materials.

You will be given a question and a set of relevant excerpts from his resumes, cover letters, and supplemental answers. Answer the question using only the information in those excerpts.

Rules:
- If the excerpts don't contain the answer, say "I don't see that in the materials provided" — don't guess.
- Cite which file each fact came from, like [cover-letter-GORGIAS.txt].
- Be concise. Use the same voice the materials use where possible.
- If multiple excerpts say different things, surface the difference rather than picking one."""


def format_context(hits):
    """Turn retrieved chunks into a formatted context block."""
    blocks = []
    for hit in hits:
        filename = hit["metadata"]["filename"].split("\\")[-1].split("/")[-1]
        blocks.append(f"--- From {filename} ---\n{hit['document']}")
    return "\n\n".join(blocks)


def answer(question, k=4):
    """Retrieve relevant chunks and get Claude to answer over them."""
    hits = retrieve(question, k=k)
    context = format_context(hits)

    user_message = f"""Question: {question}

Relevant excerpts:

{context}

Answer the question using only the excerpts above. Cite filenames in brackets."""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    return response.content[0].text, hits


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What roles have I applied for recently?"

    print(f"Question: {question}\n")
    print("=" * 60)

    response_text, hits = answer(question)

    print("\nAnswer:\n")
    print(response_text)

    print("\n" + "=" * 60)
    print(f"\nBased on {len(hits)} retrieved chunks from:")
    for hit in hits:
        filename = hit["metadata"]["filename"].split("\\")[-1].split("/")[-1]
        print(f"  - {filename} (distance: {hit['distance']:.3f})")
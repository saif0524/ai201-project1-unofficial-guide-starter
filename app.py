"""
The Unofficial Guide — UT Arlington campus survival RAG system.

A single-script Retrieval-Augmented Generation pipeline:

    documents/*.txt
        -> ingest + clean        (load_documents / clean_text)
        -> chunk                 (chunk_text)
        -> embed + store         (build_index, sentence-transformers + ChromaDB)
        -> retrieve              (retrieve)
        -> grounded generation   (generate, Groq LLM with citations)
        -> Gradio UI / CLI eval

Usage:
    python app.py                # launch the Gradio web UI (builds the index on first run)
    python app.py --rebuild      # force-rebuild the vector store from documents/
    python app.py --eval         # run the 5 evaluation questions and write eval_results.md
    python app.py --ask "..."    # one-off question from the command line
"""

import os
import re
import sys
import glob
import argparse

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

DOCS_DIR = "documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "unofficial_guide"

EMBED_MODEL = "all-MiniLM-L6-v2"          # local, 384-dim, 256-token context
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

CHUNK_SIZE = 600                          # characters (see planning.md)
CHUNK_OVERLAP = 100                       # characters
TOP_K = 5

# If the best retrieved chunk is less similar than this, we treat the query as
# unanswerable by the corpus rather than letting the model improvise. Cosine
# similarity ranges 0..1; informal student text rarely scores very high, so this
# is deliberately permissive. Tune during evaluation.
MIN_RELEVANCE = 0.20

EVAL_QUESTIONS = [
    "How do I get into a class that's already full?",
    "Where are the best quiet places to study on campus?",
    "What's the best way to get around campus without a car?",
    "What freshman mistake do upperclassmen warn against the most?",
    "Where can I find cheap food near campus?",
]


# ---------------------------------------------------------------------------
# Stage 1 — Document ingestion + cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Preprocess a raw document into clean text ready for chunking.

    - drops the synthetic-provenance tag line so it never reaches the embeddings
    - normalizes whitespace and collapses runs of blank lines to a single blank
      line (blank lines are the boundaries our chunker splits on)
    - trims trailing spaces from every line
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Skip the "[SYNTHETIC ...]" provenance marker present in the sample docs.
        if stripped.startswith("[SYNTHETIC"):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)
    # Collapse 3+ newlines into exactly two (one blank line = paragraph break).
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    """Load every .txt file in docs_dir, returning [{source, text}, ...]."""
    paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(
            f"No .txt documents found in '{docs_dir}/'. Add documents and retry."
        )
    docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        docs.append({"source": os.path.basename(path), "text": clean_text(raw)})
    return docs


# ---------------------------------------------------------------------------
# Stage 2 — Chunking
# ---------------------------------------------------------------------------
def _split_oversized(paragraph: str, size: int, overlap: int) -> list[str]:
    """Character-window split for a single paragraph longer than `size`."""
    pieces, start = [], 0
    while start < len(paragraph):
        end = start + size
        pieces.append(paragraph[start:end].strip())
        start = end - overlap            # step back to create overlap
    return [p for p in pieces if p]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into ~`size`-char chunks, respecting paragraph boundaries.

    Strategy (see planning.md): split on blank lines first so a chunk holds whole
    posts/paragraphs, then pack paragraphs together up to `size`. Carry an
    `overlap`-char tail from the previous chunk into the next so a two-sentence
    tip that straddles a boundary survives in at least one chunk.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # A paragraph that alone exceeds the size gets window-split on its own.
        if len(para) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_oversized(para, size, overlap))
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= size:
            current = candidate
        else:
            chunks.append(current.strip())
            # Seed the next chunk with the overlap tail of the one we just flushed.
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current.strip():
        chunks.append(current.strip())
    return chunks


# ---------------------------------------------------------------------------
# Stage 3 — Embedding + vector store (built lazily, cached as module globals)
# ---------------------------------------------------------------------------
_embedder = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    return _get_embedder().encode(texts, normalize_embeddings=True).tolist()


def build_index(rebuild: bool = False):
    """Create (or load) the ChromaDB collection of embedded chunks."""
    global _collection
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    # cosine space so distances map cleanly to a 0..1 similarity.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        docs = load_documents()
        ids, texts, metadatas = [], [], []
        for doc in docs:
            for i, chunk in enumerate(chunk_text(doc["text"])):
                ids.append(f"{doc['source']}::{i}")
                texts.append(chunk)
                metadatas.append({"source": doc["source"], "chunk_index": i})
        print(f"Embedding {len(texts)} chunks from {len(docs)} documents...")
        collection.add(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=_embed(texts)
        )
        print(f"Index built: {collection.count()} chunks stored in '{CHROMA_DIR}/'.")
    else:
        print(f"Loaded existing index: {collection.count()} chunks.")

    _collection = collection
    return collection


def _ensure_index():
    global _collection
    if _collection is None:
        build_index(rebuild=False)
    return _collection


# ---------------------------------------------------------------------------
# Stage 4 — Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return the top-k chunks for a query as
    [{text, source, chunk_index, similarity}, ...], best first."""
    collection = _ensure_index()
    res = collection.query(
        query_embeddings=_embed([query]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "text": text,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "similarity": round(1.0 - dist, 3),  # cosine distance -> similarity
            }
        )
    return hits


# ---------------------------------------------------------------------------
# Stage 5 — Grounded generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are The Unofficial Guide, a student-knowledge assistant for \
UT Arlington (UTA). You answer questions using ONLY the excerpts provided in the \
CONTEXT block, which are drawn from real student posts and guides.

Rules:
- Use ONLY information found in the CONTEXT. Do not add facts from your own general \
knowledge, even if you are confident they are true.
- If the CONTEXT does not contain enough information to answer, say exactly: \
"I don't have enough information in the guide to answer that." Do not guess.
- Be concise and practical, like an upperclassman giving advice.
- Cite your sources inline using the bracketed filenames from the CONTEXT, e.g. \
[01_freshman_megathread.txt]. Every claim should be traceable to a cited source."""


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[{c['source']}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def generate(query: str, chunks: list[dict]) -> str:
    """Generate a grounded answer from the retrieved chunks using Groq."""
    if not os.getenv("GROQ_API_KEY"):
        return "ERROR: GROQ_API_KEY is not set. Copy .env.example to .env and add your key."

    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    user_msg = (
        f"CONTEXT:\n{_format_context(chunks)}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the context above, citing source filenames in brackets."
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def answer(query: str, k: int = TOP_K) -> dict:
    """Full pipeline for one query. Returns answer text, sources, and the chunks."""
    chunks = retrieve(query, k=k)
    best = chunks[0]["similarity"] if chunks else 0.0

    # Grounding gate: if nothing is relevant enough, refuse rather than improvise.
    if not chunks or best < MIN_RELEVANCE:
        return {
            "answer": "I don't have enough information in the guide to answer that.",
            "sources": [],
            "chunks": chunks,
            "grounded": False,
        }

    text = generate(query, chunks)
    # Sources actually cited in the answer (deduped, retrieval order preserved),
    # falling back to all retrieved sources if the model cited none explicitly.
    seen, cited = set(), []
    for c in chunks:
        if c["source"] in text and c["source"] not in seen:
            seen.add(c["source"])
            cited.append(c["source"])
    sources = cited or sorted({c["source"] for c in chunks})
    return {"answer": text, "sources": sources, "chunks": chunks, "grounded": True}


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------
def _format_chunks_md(chunks: list[dict]) -> str:
    lines = ["### Retrieved chunks (what the answer was grounded in)\n"]
    for i, c in enumerate(chunks, 1):
        snippet = c["text"].replace("\n", " ")
        snippet = (snippet[:240] + "…") if len(snippet) > 240 else snippet
        lines.append(
            f"**{i}. `{c['source']}` (chunk {c['chunk_index']}, "
            f"similarity {c['similarity']})**\n\n> {snippet}\n"
        )
    return "\n".join(lines)


def launch_ui():
    import gradio as gr

    build_index(rebuild=False)

    def respond(question):
        if not question or not question.strip():
            return "Please enter a question.", ""
        result = answer(question.strip())
        src = ", ".join(f"`{s}`" for s in result["sources"]) or "—"
        answer_md = f"{result['answer']}\n\n**Sources:** {src}"
        return answer_md, _format_chunks_md(result["chunks"])

    with gr.Blocks(title="The Unofficial Guide — UTA") as demo:
        gr.Markdown(
            "# 🎓 The Unofficial Guide — UT Arlington\n"
            "Ask about surviving freshman year — registration, study spots, getting "
            "around, dining, dorms, and rookie mistakes. Answers are grounded in real "
            "student posts and cite their sources."
        )
        with gr.Row():
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. How do I get into a class that's already full?",
                scale=4,
            )
            ask_btn = gr.Button("Ask", variant="primary", scale=1)
        answer_out = gr.Markdown(label="Answer")
        with gr.Accordion("Show retrieved chunks", open=False):
            chunks_out = gr.Markdown()

        gr.Examples(examples=[[q] for q in EVAL_QUESTIONS], inputs=question)

        ask_btn.click(respond, inputs=question, outputs=[answer_out, chunks_out])
        question.submit(respond, inputs=question, outputs=[answer_out, chunks_out])

    demo.launch()


def run_eval():
    """Run the 5 evaluation questions and write eval_results.md."""
    build_index(rebuild=False)
    rows = []
    print("\n" + "=" * 70)
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        result = answer(q)
        print(f"\nQ{i}: {q}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources']) or '—'}")
        retrieved = ", ".join(
            f"{c['source']}#{c['chunk_index']}({c['similarity']})"
            for c in result["chunks"]
        )
        print(f"Retrieved: {retrieved}")
        rows.append((i, q, result, retrieved))
    print("\n" + "=" * 70)

    with open("eval_results.md", "w", encoding="utf-8") as f:
        f.write("# Evaluation Results (auto-generated)\n\n")
        f.write(
            "> Generated by `python app.py --eval`. Fill the **Ground-truth answer** "
            "and judge **Retrieval quality** / **Response accuracy** yourself — do not "
            "take the auto-run as a passing grade.\n\n"
        )
        for i, q, result, retrieved in rows:
            f.write(f"## Q{i}. {q}\n\n")
            f.write("**Ground-truth answer:** _[fill in from your documents]_\n\n")
            f.write(f"**System response:** {result['answer']}\n\n")
            f.write(f"**Sources cited:** {', '.join(result['sources']) or '—'}\n\n")
            f.write(f"**Chunks retrieved:** {retrieved}\n\n")
            f.write("**Retrieval quality:** Relevant / Partially relevant / Off-target\n\n")
            f.write("**Response accuracy:** Accurate / Partially accurate / Inaccurate\n\n")
            f.write("---\n\n")
    print("Wrote eval_results.md")


def main():
    parser = argparse.ArgumentParser(description="The Unofficial Guide — UTA RAG system")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the vector store")
    parser.add_argument("--eval", action="store_true", help="run the 5 eval questions")
    parser.add_argument("--ask", type=str, help="ask a single question and exit")
    args = parser.parse_args()

    if args.rebuild:
        build_index(rebuild=True)
        if not (args.eval or args.ask):
            return
    if args.eval:
        run_eval()
        return
    if args.ask:
        result = answer(args.ask)
        print("\n" + result["answer"])
        print("\nSources:", ", ".join(result["sources"]) or "—")
        return

    launch_ui()


if __name__ == "__main__":
    main()

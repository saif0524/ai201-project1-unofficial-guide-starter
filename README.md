# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Setup & Running

```bash
# 1. Install dependencies (Python 3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your Groq API key
cp .env.example .env        # then edit .env and paste your key

# 3. Build the vector store and launch the web UI
python app.py               # builds the index on first run, then opens the Gradio UI
```

Other entry points (all in the single script `app.py`):

| Command | What it does |
|---|---|
| `python app.py` | Launch the Gradio web UI |
| `python app.py --rebuild` | Rebuild the vector store from `documents/` |
| `python app.py --eval` | Run the 5 evaluation questions, write `eval_results.md` |
| `python app.py --ask "question"` | Ask one question from the command line |

The whole pipeline lives in [app.py](app.py): ingestion → chunking → embedding/vector store →
retrieval → grounded generation → interface.

---

## Domain

**Campus survival guide for UT Arlington.** The system makes searchable the practical,
unofficial know-how upperclassmen pass to freshmen — getting into full classes during
registration, the best study spots, getting around without a car, cheap food, and the rookie
mistakes to avoid. This knowledge is valuable because it's experiential and constantly updated
by students, yet scattered across subreddit megathreads, orientation wikis, and Discord
servers. Official channels describe how things are *supposed* to work; this corpus captures the
workarounds and tradeoffs students actually rely on.

---

## Document Sources

> **Provenance:** The 10 documents below are **synthetic** samples written to mirror the
> structure of real UTA student knowledge (subreddit threads, an orientation wiki, a student
> blog, a Discord export) so the pipeline could be built and tested. Each file is the corpus
> the system actually ingests. _[TODO: before final submission, replace with real collected
> sources and real URLs, or keep this disclosure.]_

**Ingestion pipeline** (`load_documents` + `clean_text` in [app.py](app.py)): every `.txt`
file in `documents/` is read, then preprocessed — the synthetic-provenance marker line is
stripped so it never reaches the embeddings, trailing whitespace is removed from each line, and
runs of 3+ newlines are collapsed to a single blank line (blank lines are the boundaries the
chunker splits on). The result is clean, structured text ready for chunking.

| # | Source | Type | File path |
|---|--------|------|-----------------|
| 1 | r/UTArlington — "Advice for incoming Mavericks" megathread | Forum thread (synthetic) | documents/01_freshman_megathread.txt |
| 2 | r/UTArlington — registration tips thread | Forum thread (synthetic) | documents/02_registration_tips.txt |
| 3 | UTA unofficial orientation wiki — getting around | Wiki page (synthetic) | documents/03_campus_transit.txt |
| 4 | r/UTArlington — best study spots thread | Forum thread (synthetic) | documents/04_study_spots.txt |
| 5 | Student blog — "things I wish I knew" | Blog post (synthetic) | documents/05_wish_i_knew.txt |
| 6 | r/UTArlington — cheap food thread | Forum thread (synthetic) | documents/06_cheap_food.txt |
| 7 | UTA Discord #class-advice export | Chat log (synthetic) | documents/07_class_advice.txt |
| 8 | r/UTArlington — common freshman mistakes thread | Forum thread (synthetic) | documents/08_freshman_mistakes.txt |
| 9 | Unofficial UTA survival guide — semester logistics | Guide page (synthetic) | documents/09_semester_logistics.txt |
| 10 | r/UTArlington — dorm/move-in advice thread | Forum thread (synthetic) | documents/10_dorm_advice.txt |

---

## Chunking Strategy

**Chunk size:** ~600 characters (≈120–150 tokens).

**Overlap:** 100 characters, carried as the tail of one chunk into the start of the next.

**Why these choices fit your documents:** The chunker (`chunk_text` in [app.py](app.py)) is
**paragraph-aware**, not a blind fixed-width split. It first splits each document on blank
lines so a chunk holds whole posts/paragraphs, then packs consecutive paragraphs together until
adding the next would exceed 600 characters. A single paragraph longer than 600 chars is
window-split with overlap as a fallback. This fits the corpus because these are short, punchy
student posts: each comment in a thread is usually one self-contained tip, so packing to ~600
chars keeps roughly one-to-three related tips per chunk — focused enough for precise retrieval,
but not so small that a two-sentence tip ("email the professor… and attend the first day")
gets fragmented. The 600-char cap also stays under the embedding model's 256-token context
window (see below) so no chunk is silently truncated. The 100-char overlap protects tips that
straddle a packing boundary.

**Final chunk count:** **37 chunks** across the 10 documents.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` — a local, 384-dimensional
model run with cosine similarity in ChromaDB. Chosen because it's free, runs on CPU with no API
latency, and performs well on short, informal English text like student posts. Embeddings are
normalized so ChromaDB's cosine distance maps cleanly to a 0–1 similarity score, which the
retrieval gate uses.

**Production tradeoff reflection:** For real users with no cost constraint I'd weigh:
- **Context length:** MiniLM truncates at **256 tokens**, which is the main reason chunks are
  capped at ~600 chars. A longer-context embedder (e.g. `bge-large-en-v1.5`, OpenAI
  `text-embedding-3-large`, Voyage) would let me embed whole posts or even whole short docs,
  reducing boundary-split failures like the one in the Failure Case below.
- **Accuracy on domain text:** campus slang, building abbreviations (ERB, CAPPA), and product
  names (MyMav, Mav Mover) are near out-of-vocabulary for a small general model — a larger or
  domain-adapted model would retrieve them more reliably.
- **Multilingual:** MiniLM is English-centric; an international student body posting in other
  languages would justify a multilingual model.
- **Latency / local vs. API:** the local model adds zero per-query cost and keeps
  campus-specific content private, but a hosted API model trades that for higher accuracy and
  offloaded compute. For a small, privacy-sensitive student tool, local is the right default;
  at scale I'd benchmark a hosted model on a labeled retrieval set before switching.

---

## Grounded Generation

Grounding is enforced by **three layers** — a relevance gate, the context format, and the
system prompt — not by the prompt alone.

**1. Retrieval gate (structural).** Before any LLM call, `answer()` checks the cosine
similarity of the best-retrieved chunk. If it falls below `MIN_RELEVANCE = 0.20`, the system
short-circuits and returns *"I don't have enough information in the guide to answer that"*
without calling the model at all — so an off-topic query (e.g. "Who won the 2026 World Cup?")
can't trigger a hallucinated answer.

**2. Context format (structural).** Retrieved chunks are concatenated into a `CONTEXT` block,
each prefixed with its source filename in brackets, e.g. `[02_registration_tips.txt]`. The
model never sees raw documents — only the top-k retrieved chunks — so it has nothing to draw on
beyond what retrieval surfaced.

**3. System prompt grounding instruction (the actual text):**

> *"You answer questions using ONLY the excerpts provided in the CONTEXT block… Use ONLY
> information found in the CONTEXT. Do not add facts from your own general knowledge, even if
> you are confident they are true. If the CONTEXT does not contain enough information to
> answer, say exactly: 'I don't have enough information in the guide to answer that.' Do not
> guess… Cite your sources inline using the bracketed filenames from the CONTEXT."*

Generation temperature is set to `0.2` to keep answers close to the source text.

**How source attribution is surfaced in the response:** The model cites bracketed filenames
inline next to each claim. After generation, `answer()` scans the response for which source
filenames actually appear, dedupes them, and surfaces them as a **Sources** list under the
answer in the UI. The UI also has a "Show retrieved chunks" panel exposing every retrieved
chunk with its source, chunk index, and similarity score — so a grader can verify the answer is
traceable to real retrieved text.

---

## Evaluation Report

Run with `python app.py --eval` (full transcript with retrieved chunk IDs + similarity scores
is written to `eval_results.md`). Summary of the run on the synthetic corpus:

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | How do I get into a class that's already full? | Waitlist in MyMav, email prof for an override, attend first day, swap don't drop | Listed all four tactics, cited `02_registration_tips.txt` + `05_wish_i_knew.txt`; top chunk sim 0.68 | Relevant | Accurate |
| 2 | Where are the best quiet places to study on campus? | Upper floors of Central Library, CAPPA studios, Science & Eng. Library, UC 2nd floor | Named all four, cited `04_study_spots.txt`; top chunk sim 0.75 | Relevant | Accurate |
| 3 | Best way to get around campus without a car? | Maverick Shuttle/Mav Mover, walking (~15 min), biking, rideshare | Gave shuttle + walking + biking + living on campus, cited `03_campus_transit.txt` + `10_dorm_advice.txt`; top sim 0.71 | Relevant | Accurate |
| 4 | What freshman mistake do upperclassmen warn against most? | Buying a parking permit they don't need | Correctly identified the parking-permit mistake, cited `08_freshman_mistakes.txt` + `05_wish_i_knew.txt`; top sim 0.57 | Relevant | Accurate |
| 5 | Where can I find cheap food near campus? | Cooper St / UTA Blvd strip, Mav Express markets, pho/ramen, student discounts | Listed all of these, cited `06_cheap_food.txt`; top sim 0.72 | Relevant | Accurate |

**Retrieval quality:** Relevant (5/5) — for the planned questions the correct source document
ranked first every time.
**Response accuracy:** Accurate (5/5) on the planned questions; answers stayed grounded and
cited correctly. _Note: these 5 questions map cleanly onto single source documents, which is
why they score perfectly. The Failure Case below shows where this breaks down._

---

## Failure Case Analysis

**Question that failed:** *"What time does the shuttle stop running at night?"*

**What the system returned:** *"I don't have enough information in the guide to answer that."*
The five chunks retrieved were `01_freshman_megathread`, `02_registration_tips`,
`04_study_spots`, `07_class_advice`, and `01_freshman_megathread` — **none from the transit
document** (`03_campus_transit.txt`), even though that's the only doc about the shuttle.

**Root cause (retrieval stage, compounded by a corpus-coverage gap).** Two things combined:
1. **Retrieval miss.** The query's strongest terms — "time," "night," "running" — semantically
   matched the *scheduling/timing* language in registration ("enrollment appointment time,"
   "turn over fast in the first hour") and the "late afternoon / at night" phrasing in the
   study-spots doc more strongly than the transit chunk, which describes shuttle *frequency*
   ("runs frequently during the day… less often at night") without the words the query
   emphasized. So the relevant chunk never made the top-5.
2. **Coverage gap.** Even with perfect retrieval, the corpus never states an actual shuttle
   *stop time* — it only says service is reduced at night. The honest answer genuinely isn't in
   the documents, so the grounding gate's refusal is arguably the *correct* behavior — but it
   refused for the wrong reason (off-target retrieval), not because it located the transit doc
   and found it lacked a specific time.

**What you would change to fix it:** (a) Add **hybrid retrieval** — a keyword/BM25 pass merged
with the semantic search — so a chunk that literally contains "shuttle" and "night" is surfaced
even when the dense embedding ranks it lower. (b) Raise `top_k` or add a small re-ranker so the
transit chunk has a chance to appear. (c) Close the coverage gap by adding a source with
explicit shuttle hours. The deeper lesson: retrieval quality on this corpus is fragile for
questions whose *wording* doesn't align with how students phrased the advice.

---

## Spec Reflection

_[TODO: confirm these reflect your experience and edit in your own voice.]_

**One way the spec helped you during implementation:** The Chunking Strategy in `planning.md`
(≈600 chars, paragraph-aware, 100-char overlap, justified by the short-post structure of the
corpus) translated almost directly into the `chunk_text` function — the spec already named the
size, the overlap, and the "split on blank lines first" rule, so implementation was a matter of
encoding decisions that were already made rather than rediscovering them mid-code. Writing the
embedding-model tradeoffs in advance also meant the 256-token context limit was a known
constraint that *set* the chunk size, instead of a bug discovered later.

**One way your implementation diverged from the spec, and why:** The spec described retrieval
as "embed query → top-k=5" with no notion of a relevance threshold. During implementation it
became clear that off-topic queries would still return five (irrelevant) chunks and invite a
hallucinated answer, so I added a `MIN_RELEVANCE` gate that refuses to call the LLM when the
best chunk's similarity is too low. This wasn't in the plan because the failure mode only became
obvious once the system could actually be queried.

---

## AI Usage

_[TODO: this is an accurate record of how AI was used — verify it, and add anything you
directed or overrode yourself.]_

**Instance 1 — generating the document corpus**

- *What I gave the AI:* My chosen domain (UTA campus survival guide) and the subtopics I wanted
  covered (registration, transit, dining, study spots, dorms, mistakes).
- *What it produced:* 10 synthetic documents mimicking real student-post structure (subreddit
  threads, a wiki page, a blog, a Discord export), each tagged as synthetic.
- *What I changed or overrode:* _[your input — e.g. which subtopics you added/removed, whether
  you later swapped in real sources, the decision to keep them synthetic and disclose it.]_

**Instance 2 — implementing the chunker and grounding**

- *What I gave the AI:* My `planning.md` Chunking Strategy section and the Grounded Generation
  requirements.
- *What it produced:* A paragraph-aware `chunk_text` (split on blank lines, pack to ~600 chars,
  overlap) and a three-layer grounding design (relevance gate + bracketed-source context format
  + a strict system prompt).
- *What I changed or overrode:* _[your input — e.g. tuning `MIN_RELEVANCE`, the chunk size, the
  top-k value, or the system-prompt wording after seeing eval results.]_

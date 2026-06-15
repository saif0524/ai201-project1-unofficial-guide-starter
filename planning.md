# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

> **DRAFT NOTE (delete before submission):** Sections below are AI-drafted starting points.
> Verify each against your own judgment and your actual documents. Items marked **[TODO: you]**
> require your own work — especially document collection and expected answers.

---

## Domain

**Campus survival guide for UT Arlington** — the practical, unofficial know-how that
upperclassmen pass to freshmen: how to actually get into full classes during registration,
which buildings have the good study spots, how to get around without a car, where to find
cheap food, and the rookie mistakes to avoid. This knowledge is valuable because it's
experiential and constantly updated by students, yet it's scattered across subreddit
megathreads, orientation wikis, and Discord servers — never collected in one searchable place.
Official channels (the course catalog, the orientation handbook, the university website)
describe how things are *supposed* to work, not the workarounds and tradeoffs students
actually rely on.

---

## Documents

> **PROVENANCE NOTE:** The 10 files below are **synthetic** sample documents generated to
> develop and test the pipeline (each file is tagged `[SYNTHETIC ...]` in its header). They
> cover the right subtopics — registration, transit, dining, studying, dorms, mistakes — so the
> retrieval/generation pipeline can be built now.
> **[TODO: you]** Before submission, either replace these with real UTA sources (paste actual
> post text into the same filenames and add real URLs to the table) or disclose in the AI Usage
> section that the corpus is synthetic.

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | r/UTArlington | "Advice for incoming freshmen" megathread | documents/01_freshman_megathread.txt |
| 2 | r/UTArlington | Course registration tips thread | documents/02_registration_tips.txt |
| 3 | Orientation wiki / unofficial FAQ | Getting-around-campus guide | documents/03_campus_transit.txt |
| 4 | r/UTArlington | Best study spots discussion | documents/04_study_spots.txt |
| 5 | Student blog / Medium post | "What I wish I knew freshman year" | documents/05_wish_i_knew.txt |
| 6 | r/UTArlington | Cheap/good food near campus thread | documents/06_cheap_food.txt |
| 7 | Discord / forum export | Class selection advice | documents/07_class_advice.txt |
| 8 | r/UTArlington | Common freshman mistakes thread | documents/08_freshman_mistakes.txt |
| 9 | Unofficial survival guide PDF/site | Semester logistics & deadlines | documents/09_semester_logistics.txt |
| 10 | r/UTArlington | Dorm/housing move-in advice | documents/10_dorm_advice.txt |

---

## Chunking Strategy

> **Starting draft — confirm after you skim your actual documents in Milestone 2.**
> Survival-guide corpora are mixed: orientation wikis/FAQs are long and topically structured,
> while subreddit threads are many short posts strung together. A moderate chunk with light
> overlap balances both.

**Chunk size:** ~600 characters (roughly 120–150 tokens), with paragraph/post boundaries
respected where possible (split on blank lines first, then pack up to the size limit).

**Overlap:** ~100 characters. Survival advice often spans the sentence that *names* a thing
("registration opens at your enrollment appointment time") and the sentence that gives the
*tip* ("so set an alarm and refresh") — overlap keeps that pairing intact across a boundary.

**Reasoning:** A small chunk keeps a single retrieved result focused on one piece of advice
(better precision for specific questions), and stays well under the embedding model's 256-token
limit (see below) so nothing gets silently truncated. Overlap guards against splitting a
two-sentence tip across chunks. **[TODO: you]** Adjust the size once you see how long your
real posts are — if your sources are mostly short reviews, drop toward ~400; if they're long
guides, consider ~800.

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers` (already in
`requirements.txt`). It's local (no API cost or latency to a provider), fast, and produces
384-dimensional vectors that work well for short, informal English text like student posts.

**Top-k:** 5. Survival questions ("how do I avoid freshman mistakes?") often have several
distinct valid answers spread across sources, so retrieving 5 chunks gives the generator
enough material to synthesize without flooding the context with noise. **[TODO: you]** Tune
between 3–6 during evaluation.

**Production tradeoff reflection:** `all-MiniLM-L6-v2` has only a **256-token context window**,
which caps how large a chunk can be before truncation — that's the main constraint driving the
small chunk size above. If cost weren't a concern, I'd weigh: (1) a larger/stronger model like
`bge-large-en-v1.5` or an API model (OpenAI `text-embedding-3-large`, Voyage) for higher
retrieval accuracy and longer context, at the cost of latency and per-call price; (2)
**multilingual** support if international students post in other languages — MiniLM is
English-centric; (3) **domain accuracy** — campus slang, professor nicknames, and building
abbreviations are effectively out-of-vocabulary, so a model fine-tuned on conversational/web
text would likely retrieve better than one trained on formal corpora; (4) **local vs. API** —
local keeps student data private and avoids rate limits, which matters for a tool handling
campus-specific content.

---

## Evaluation Plan

> Expected answers below are derived from the current (synthetic) corpus. **[TODO: you]**
> Re-confirm them if you swap in real documents.

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students recommend doing to get into a class that's already full? | Join the MyMav waitlist (it moves fast during add/drop), email the professor before the semester for an override/add permit, attend the first lecture in person, and use "swap" rather than dropping. (docs 01, 02, 05, 09) |
| 2 | Which spots on campus do students recommend for quiet, focused studying? | The upper/top floors of the Central Library (quiet zones; lower floors are social), the CAPPA architecture studios at night, the Science & Engineering Library, and reservable library study rooms for groups. (docs 04, 05) |
| 3 | What's the best way to get around campus without a car? | Arlington has no city bus system; use the free Maverick Shuttle / "Mav Mover," the on-demand rideshare, biking, or just walking (campus is ~15 min end to end). A car/permit isn't needed freshman year, especially on campus. (docs 03, 10) |
| 4 | What freshman mistake do upperclassmen most often warn against? | Buying a parking permit they don't need (top recurring answer); also not checking MyMav for holds, skipping office hours, overloading credits, and treating meal swipes as unlimited. (docs 01, 05, 08) |
| 5 | Where do students go for cheap food near campus? | The Cooper St / UTA Blvd strip (taquerias, fast food under $10), on-campus Mav Express markets (dining dollars), nearby pho/ramen spots, student-discount spots (show Mav ID), and free food at club events. (docs 06) |

---

## Anticipated Challenges

1. **Time-sensitive, campus-specific advice.** Survival tips go stale — a registration trick
   or a "best food" recommendation from three years ago may be wrong now, but the embedding
   has no concept of recency. Retrieval will happily return outdated advice with the same
   confidence as current advice. Mitigation: prefer recent sources when collecting, and
   consider noting source dates in chunk metadata.

2. **Topic-mixing in long threads + boundary splits.** Megathreads jam many unrelated tips
   into one document. If chunks span topic shifts, a single chunk blends registration and
   dining advice, hurting retrieval precision; conversely, a single piece of advice can split
   across a chunk boundary, so retrieval returns only half of it. Mitigation: split on
   post/paragraph boundaries before packing to size, and use overlap.

---

## Architecture

```
  ┌──────────────────┐
  │ documents/*.txt   │   Raw student posts, threads, guides (collected by hand)
  └─────────┬─────────┘
            │  Document Ingestion  (Python stdlib / pdfplumber for any .pdf)
            ▼
  ┌──────────────────┐
  │  Chunking         │   ~600 chars, ~100 overlap, split on blank lines first
  └─────────┬─────────┘
            │
            ▼
  ┌──────────────────┐
  │ Embedding +       │   sentence-transformers (all-MiniLM-L6-v2) → 384-dim vectors
  │ Vector Store      │   stored in ChromaDB (local, persisted to chroma_db/)
  └─────────┬─────────┘
            │
            ▼
  ┌──────────────────┐
  │  Retrieval        │   embed query → ChromaDB similarity search → top-k=5 chunks
  └─────────┬─────────┘
            │
            ▼
  ┌──────────────────┐
  │  Generation       │   Groq API (Llama model) — grounded prompt + retrieved chunks
  └─────────┬─────────┘   → cited, grounded answer
            ▼
   User-facing answer with sources
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**
Tool: Claude (Claude Code). Input: the **Chunking Strategy** section above plus 1–2 sample
documents from `documents/`. Ask it to implement `load_documents()` (read every file in
`documents/`, return text + filename) and `chunk_text(text, size=600, overlap=100)` that splits
on blank lines before packing to size. Verify: print the first 3 chunks and the total chunk
count, confirm no chunk exceeds ~600 chars and boundaries land sensibly.

**Milestone 4 — Embedding and retrieval:**
Tool: Claude. Input: the **Retrieval Approach** section. Ask it to embed all chunks with
`all-MiniLM-L6-v2`, store them in a persisted ChromaDB collection (with filename metadata), and
implement `retrieve(query, k=5)`. Verify: run my 5 evaluation questions and eyeball whether the
returned chunks are on-topic; check the collection count equals my chunk count.

**Milestone 5 — Generation and interface:**
Tool: Claude. Input: the **Grounded Generation** requirements + my retrieval function. Ask it to
build a Groq prompt that injects the retrieved chunks and instructs the model to answer *only*
from them and cite source filenames, refusing when the context doesn't contain the answer.
Verify: ask a question my corpus can't answer and confirm the system declines rather than
hallucinating; spot-check that citations match the chunks actually retrieved.

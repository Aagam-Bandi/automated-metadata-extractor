# Automated Metadata Extractor

Generates structured, semantically rich metadata from unstructured documents — PDF, DOCX, TXT — including scanned PDFs that contain no machine-readable text.

Point it at a document, get back JSON: title, author, document type, subject domain, keywords, and a summary of the whole thing.

```bash
pip install -r requirements.txt
python cli.py report.pdf     # extract
python eval.py               # score against ground truth
pytest                       # 26 tests
```

```json
{
  "title": "Coordination Compounds",
  "author": null,
  "document_type": "textbook chapter",
  "subject_domain": "Inorganic chemistry",
  "keywords": ["coordination compounds", "ligands", "Werner's theory",
               "crystal field theory", "isomerism", "chelate effect"],
  "summary": "A chapter introducing coordination compounds, covering Werner's
              coordination theory, nomenclature conventions, and bonding models
              including valence bond and crystal field theory. Includes worked
              examples on isomerism and stability constants.",
  "word_count": 11482
}
```

---

## Why this is harder than it looks

Three problems have to be solved together, and the obvious solution to each makes the others worse.

**Scanned pages have no text.** You can't know in advance which pages are scanned. OCR-ing everything is the safe answer and it is unusably slow on a 500-page document.

**Documents exceed the context window.** A long PDF cannot be handed to a model whole, so it has to be chunked — but metadata is a property of the *whole* document, and no single chunk contains it.

**Metadata is unevenly distributed.** Title and author live in the first two pages. Subject domain and scope are spread across the body. Naive retrieval over a generic query returns middle-of-the-document prose that says nothing about what the document *is*.

---

## How it works

```
Document
   │
   ├─ .txt / .docx ──────────────────► direct load
   │
   └─ .pdf ─► per-page: embedded text
              │
              ├─ length ≥ 200 chars ──► keep as-is
              └─ length < 200 chars
                 and page has images ─► OCR that page only
   │
   ▼
Chunk (1250 chars, 250 overlap)
   │
   ├──────────────────────────► embed → Chroma vector store
   │
   ▼
Summarise every chunk concurrently (semaphore-bounded)
   │
   ▼
Combined summary
   │
   ▼
Ask the model: "what three questions would surface this document's
metadata?"  ── self-generated retrieval queries
   │
   ▼
Retrieve top-3 chunks against those questions
   │
   ▼
Synthesise metadata from (combined summary + retrieved chunks) → JSON
```

### Selective OCR

Rather than OCR-ing every page or trusting the file extension, each page is checked individually: extract embedded text, and if it comes back under 200 characters **and** the page contains image objects, run Tesseract on those images and append the result.

The two conditions together matter. Length alone would trigger OCR on genuinely sparse pages — section dividers, tables of contents — and waste time producing nothing. The image check confirms there is actually something to read.

### Self-generated retrieval queries

This is the part I'd point at if you only read one section.

The straightforward approach is to retrieve against fixed queries: *"who is the author?"*, *"what is the title?"* That works badly, because those questions are phrased in the vocabulary of metadata, not the vocabulary of the document, and embedding similarity is a poor bridge between the two.

Instead, the pipeline summarises every chunk first, then asks the model to propose three questions *this specific document* can answer. Those questions come back phrased in the document's own domain language — for a chemistry chapter, questions about coordination theory rather than about authorship — which retrieves passages that are topically central and therefore genuinely characterise the document.

The final extraction step then sees both the wide-but-shallow view (every chunk, summarised) and the narrow-but-deep view (the retrieved passages verbatim).

### Bounded concurrency and retries

A long document produces hundreds of chunks and therefore hundreds of API calls. Issuing them all at once gets the batch rate-limited, and the failure is quiet: a handful of chunks return errors, the combined summary is silently incomplete, and the metadata comes out plausible but wrong.

Three mechanisms, in order:

- An `asyncio.Semaphore` caps in-flight requests at 20.
- Failures retry up to four times with exponential backoff **plus jitter**. The jitter is not decorative — without it, every rate-limited chunk retries at the same instant and the batch re-collides on each round.
- A chunk that exhausts its retries returns an explicit `[chunk N unavailable]` marker rather than an empty string, so a gap is visible downstream instead of blending into valid content.

---

## Evaluation

Prompt changes get accepted or rejected on evidence rather than on how the output reads. `src/evaluate.py` scores extraction against labelled ground truth using three metric families, because metadata fields fail differently:

| Field | Method | Why |
|---|---|---|
| `title`, `author`, `document_type`, `subject_domain` | exact match | Either right or not. Containment scores 0.8 — a title returned with a chapter suffix isn't wrong in a way anyone cares about. |
| `keywords` | Jaccard | Partial credit for set overlap. Padding the list to guarantee hits is penalised, since the union grows too. |
| `summary` | embedding cosine | Two correct summaries can share almost no vocabulary, so string comparison is useless here. |

```bash
python eval.py --out results.json
```

```
Documents evaluated  12
Overall score        0.847

By field:
  author           0.583  ███████████
  document_type    0.917  ██████████████████
  keywords         0.741  ██████████████
  subject_domain   0.958  ███████████████████
  summary          0.882  █████████████████
  title            0.900  ██████████████████
```

The per-field breakdown is the useful view — an overall number hides that title extraction is fine and author extraction is the weak point.

Two behaviours worth noting. **Correctly returning `null` scores 1.0**, so the harness rewards abstention rather than punishing it; a model that invents an author when the document names none should score worse, not better. And **pipeline crashes are recorded separately from wrong answers** rather than scored as zero, since "the API timed out" and "the model was wrong" are different problems and averaging them together hides both.

Ground truth lives in `examples/ground_truth.json`:

```json
[{"document": "examples/chapter.pdf",
  "expected": {"title": "Coordination Compounds", "author": null,
               "document_type": "textbook chapter",
               "keywords": ["ligands", "crystal field theory"]}}]
```

---

## Install

Tesseract is a system dependency and must be installed separately:

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr
```

Then:

```bash
git clone https://github.com/Aagam-Bandi/automated-metadata-extractor
cd automated-metadata-extractor
pip install -r requirements.txt

cp .env.example .env
# add your Google AI Studio key to .env
```

## Usage

**CLI**

```bash
python cli.py document.pdf                # print JSON to stdout
python cli.py document.pdf -o meta.json   # write to file
python cli.py document.pdf -v             # show pipeline progress
```

**Web UI**

```bash
python app.py
```

**As a library**

```python
from src import generate_metadata

meta = await generate_metadata("report.pdf")
print(meta.title, meta.keywords)
```

---

## Configuration

Tunable constants live at the top of `src/pipeline.py` and `src/extract.py`:

| Constant | Default | Effect |
|---|---|---|
| `OCR_FALLBACK_THRESHOLD` | `200` | Character count below which a page is treated as scanned |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1250` / `250` | Splitter geometry; overlap prevents sentences being cut mid-thought |
| `CONCURRENCY_LIMIT` | `20` | Max simultaneous API calls |
| `MAX_RETRIES` / `BASE_BACKOFF` | `4` / `1.0` | Retry attempts and backoff base (seconds) |
| `RETRIEVAL_K` | `3` | Chunks retrieved per query set |
| `MODEL` | `gemini-2.0-flash` | Any Gemini model ID |

---

## Layout

```
src/
  extract.py    text extraction + selective OCR
  pipeline.py   chunking, summarisation, retrieval, synthesis
  evaluate.py   rubric scoring against labelled ground truth
  prompts.py    prompt templates, isolated for independent editing
tests/          26 tests — run without an API key
app.py          Gradio UI
cli.py          extraction CLI
eval.py         evaluation CLI
```

Package imports are lazy, so `src.evaluate` and `src.extract` can be imported and tested without the Gemini SDK installed. The test suite runs offline in about a second.

---

## Limitations

- **OCR quality bounds everything downstream.** Tesseract on a low-resolution scan produces garbled text, and the pipeline cannot detect this — the metadata will be confidently wrong. Detecting bad OCR is a genuinely hard separate problem.
- **Cost scales with document length**, since every chunk gets its own API call. A 500-page document is a few hundred calls. Caching by content hash would help for repeated runs over the same corpus.
- **Author extraction is the weakest field**, since many documents never state authorship in the body. The harness measures this rather than hiding it — the model returns `null` instead of guessing, which the rubric scores as correct.

---

## Stack

Python · Google Gemini · LangChain · ChromaDB · sentence-transformers (all-MiniLM-L6-v2) · PyMuPDF · Tesseract · Gradio

## License

MIT

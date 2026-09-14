"""Metadata generation pipeline: chunk -> summarise -> self-query -> retrieve -> extract."""

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, asdict

import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .extract import extract, ExtractionResult
from . import prompts

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1250
CHUNK_OVERLAP = 250
CONCURRENCY_LIMIT = 20
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_K = 3
MODEL = "gemini-2.0-flash"


@dataclass
class Metadata:
    """Structured output. Kept as a dataclass so callers get a stable contract
    even when the model's raw JSON drifts."""
    title: str | None = None
    author: str | None = None
    document_type: str | None = None
    subject_domain: str | None = None
    keywords: list[str] | None = None
    summary: str | None = None
    word_count: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _configure() -> None:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    genai.configure(api_key=key)


MAX_RETRIES = 4
BASE_BACKOFF = 1.0


async def _summarise_chunk(model, text: str, idx: int, sem: asyncio.Semaphore) -> str:
    """Summarise one chunk, retrying transient failures with jittered backoff.

    Rate limits and 5xx responses are transient and worth retrying; a malformed
    request is not, but distinguishing them reliably across SDK versions is
    fragile, so all errors retry and the attempt count bounds the cost.

    Jitter matters: without it, every rate-limited chunk retries at the same
    instant and the batch re-collides on each round.

    After the final attempt the chunk returns a visible marker rather than
    raising. One bad chunk should not discard the document, but it must not
    blend into valid content either — a silently missing chunk produces a
    summary that looks complete and isn't.
    """
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await model.generate_content_async(
                    prompts.CHUNK_SUMMARY.format(chunk=text)
                )
                return resp.text.strip()
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    logger.error("chunk %d failed after %d attempts: %s",
                                 idx, MAX_RETRIES, exc)
                    return f"[chunk {idx} unavailable: {type(exc).__name__}]"
                delay = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("chunk %d attempt %d failed (%s), retrying in %.1fs",
                               idx, attempt + 1, type(exc).__name__, delay)
                await asyncio.sleep(delay)


def _parse_json(raw: str) -> dict:
    """Strip markdown fences the model sometimes emits despite instructions."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.removeprefix("json").strip()
    if text.endswith("```"):
        text = text[: text.rfind("```")].strip()
    return json.loads(text)


async def generate_metadata(path: str) -> Metadata:
    """Run the full pipeline over a single document."""
    _configure()

    logger.info("extracting %s", path)
    result: ExtractionResult = extract(path)
    if result.pages_total:
        logger.info("%d/%d pages required OCR", result.pages_ocred, result.pages_total)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(result.documents)
    logger.info("split into %d chunks", len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma.from_documents(chunks, embeddings)
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVAL_K})

    model = genai.GenerativeModel(MODEL)
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    summaries = await asyncio.gather(*[
        _summarise_chunk(model, c.page_content, i, sem) for i, c in enumerate(chunks)
    ])
    combined = "\n".join(summaries)

    # Self-querying step: the document's own summary is used to generate the
    # retrieval queries. Rather than guessing at fixed questions ("who wrote
    # this?"), the model proposes questions this specific document can answer,
    # which retrieves chunks that are actually metadata-bearing.
    q_resp = model.generate_content(prompts.RETRIEVAL_QUESTIONS.format(summaries=combined))
    questions = q_resp.text.strip()
    logger.info("generated retrieval questions:\n%s", questions)

    context = "\n".join(d.page_content for d in retriever.invoke(questions))

    final = model.generate_content(prompts.METADATA.format(
        summaries=combined, context=context, word_count=result.word_count
    ))

    try:
        data = _parse_json(final.text)
    except json.JSONDecodeError as exc:
        logger.error("model returned unparseable JSON: %s", exc)
        raise ValueError("metadata extraction returned malformed JSON") from exc

    known = {f for f in Metadata.__dataclass_fields__}
    meta = Metadata(**{k: v for k, v in data.items() if k in known})
    meta.word_count = result.word_count
    return meta


def generate_metadata_sync(path: str) -> Metadata:
    """Blocking wrapper for scripts and the Gradio UI."""
    return asyncio.run(generate_metadata(path))

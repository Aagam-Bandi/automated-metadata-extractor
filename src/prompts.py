"""Prompt templates, kept separate so they can be edited and diffed independently."""

CHUNK_SUMMARY = """Summarise the following excerpt in 1-2 sentences.

Preserve any metadata-bearing details verbatim if present: document title,
author names, dates, publisher, section headings, document type. These are
needed downstream even if they seem incidental to the excerpt's content.

Excerpt:
---
{chunk}
---"""


RETRIEVAL_QUESTIONS = """Below is a sequence of summaries covering every chunk of one document.

Propose exactly three questions that this document can answer and whose answers
would surface the passages richest in metadata (authorship, title, purpose,
scope, domain, publication context).

Return only the questions, one per line, with no numbering or preamble.

Summaries:
{summaries}"""


METADATA = """You are a document analyst. Synthesise metadata from two sources:
a chunk-by-chunk summary of the whole document, and specific excerpts retrieved
as most likely to carry metadata.

Return a single JSON object and nothing else. No preamble, no markdown fences,
no text before or after the braces.

Required keys:
  title            string or null
  author           string or null
  document_type    string  (e.g. "textbook chapter", "research paper", "report")
  subject_domain   string
  keywords         array of 5-10 strings
  summary          string, 3-4 sentences describing the document as a whole

Set a key to null rather than guessing when the sources do not support a value.

SOURCE 1 - CHUNK SUMMARIES
{summaries}

SOURCE 2 - RETRIEVED EXCERPTS
{context}

The document's word count is {word_count}; you do not need to return it."""

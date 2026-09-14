"""Text extraction from PDF, DOCX and TXT, with selective OCR for scanned PDFs."""

import io
import logging
import os
from dataclasses import dataclass

import pymupdf
import pytesseract
from PIL import Image
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx", ".doc", ".txt"}

# A page yielding fewer than this many characters of embedded text is treated
# as scanned, and its images are sent to OCR. Tuned empirically: real text
# pages in the corpora tested cleared this comfortably, while scanned pages
# returned near-zero.
OCR_FALLBACK_THRESHOLD = 200


@dataclass
class ExtractionResult:
    documents: list[Document]
    word_count: int
    pages_ocred: int
    pages_total: int


def _extract_pdf(path: str) -> tuple[str, int, int]:
    """Return (text, pages_ocred, pages_total) for a PDF.

    OCR is applied per page rather than to the whole document. Running
    Tesseract over every page of a text-native PDF is slow and adds nothing,
    so we only fall back when a page's embedded text comes back suspiciously
    short *and* the page contains images.
    """
    doc = pymupdf.open(path)
    pages: list[str] = []
    ocred = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        images = page.get_images(full=True)

        if images and len(page_text) < OCR_FALLBACK_THRESHOLD:
            ocred += 1
            logger.info("page %d: %d image(s), text below threshold — running OCR",
                        page_num + 1, len(images))
            for idx, img in enumerate(images):
                try:
                    base = doc.extract_image(img[0])
                    image = Image.open(io.BytesIO(base["image"]))
                    page_text += "\n" + pytesseract.image_to_string(image)
                except Exception as exc:
                    logger.warning("page %d image %d failed OCR: %s",
                                   page_num + 1, idx + 1, exc)

        pages.append(page_text)

    total = len(doc)
    doc.close()
    return "\n\n".join(pages), ocred, total


def extract(path: str) -> ExtractionResult:
    """Load a document into LangChain Documents, OCR-ing scanned PDF pages."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise ValueError(f"unsupported file type {ext!r}; expected one of {sorted(SUPPORTED)}")

    ocred = total = 0

    if ext == ".txt":
        documents = TextLoader(path).load()
    elif ext in {".docx", ".doc"}:
        documents = UnstructuredWordDocumentLoader(path).load()
    else:
        text, ocred, total = _extract_pdf(path)
        documents = [Document(
            page_content=text,
            metadata={"source": os.path.basename(path)},
        )]

    word_count = sum(len(d.page_content.split()) for d in documents)
    return ExtractionResult(documents, word_count, ocred, total)

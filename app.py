"""Gradio interface for the metadata extractor."""

import logging

import gradio as gr
from dotenv import load_dotenv

from src import generate_metadata_sync

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")


def _handle(file):
    if file is None:
        return {"error": "Upload a document to begin."}
    try:
        return generate_metadata_sync(file.name).to_dict()
    except Exception as exc:
        logging.exception("extraction failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def build():
    with gr.Blocks(theme=gr.themes.Soft(), title="Automated Metadata Extractor") as demo:
        gr.Markdown(
            "# Automated Metadata Extractor\n"
            "Upload a `.pdf`, `.docx` or `.txt`. Scanned pages are OCR-ed automatically."
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(label="Document", file_types=[".pdf", ".docx", ".txt"])
                run = gr.Button("Generate metadata", variant="primary")
            with gr.Column(scale=2):
                out = gr.JSON(label="Metadata")
        run.click(fn=_handle, inputs=file_in, outputs=out)
    return demo


if __name__ == "__main__":
    build().launch()

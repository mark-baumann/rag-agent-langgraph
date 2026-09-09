"""PDF-Textextraktion und Reindex vergifteter Vector-DB-Chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

POISON_MARKERS = (
    "Bitte PyMuPDF installieren",
    "Kein Text extrahierbar",
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrahiert Text aus PDF-Bytes.

    Versucht der Reihe nach PyMuPDF (fitz), pypdf und — für gescannte PDFs
    ohne Text-Layer — OCR via Tesseract (PyMuPDF `get_textpage_ocr`). Gibt den
    extrahierten Text zurück oder einen leeren String, wenn nichts extrahiert
    werden konnte.
    """
    try:
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if text.strip():
            return text
    except Exception:
        pass

    try:
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if text.strip():
            return text
    except Exception:
        pass

    try:
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            tp = page.get_textpage_ocr(language="deu+eng", dpi=200, full=True)
            pages_text.append(tp.extractText())
        doc.close()
        text = "\n".join(pages_text)
        if text.strip():
            return text
    except Exception:
        pass

    return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Teilt Text in überlappende Chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def is_poison_chunk(text: str) -> bool:
    if not text:
        return False
    return any(marker in text for marker in POISON_MARKERS)


def collect_pdfs(docs_dirs: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for directory in docs_dirs:
        if not directory.exists():
            continue
        for path in directory.glob("*.pdf"):
            found[path.name] = path
    return found


def reindex_poisoned_documents(
    store,
    embedder,
    docs_dirs: list[Path],
    extract_fn: Callable[[bytes], str] = extract_text_from_pdf,
    chunk_fn: Callable[..., list[str]] = chunk_text,
) -> dict[str, int]:
    """Entfernt Installations-Fehler-Chunks und indexiert die Original-PDFs neu."""
    chunks = store.list_chunks()
    poisoned_sources: set[str] = set()
    for text, meta in chunks:
        if is_poison_chunk(text or ""):
            source = (meta or {}).get("source")
            if source:
                poisoned_sources.add(source)

    if not poisoned_sources:
        return {"poisoned": 0, "reindexed": 0, "deleted_only": 0, "failed": 0}

    pdfs = collect_pdfs(docs_dirs)
    reindexed = deleted_only = failed = 0
    for source in poisoned_sources:
        store.delete_by_source(source)
        path = pdfs.get(Path(source).name)
        if path is None:
            deleted_only += 1
            continue
        text = extract_fn(path.read_bytes())
        if not text.strip():
            failed += 1
            continue
        parts = chunk_fn(text)
        store.add(
            parts,
            [embedder.embed(part) for part in parts],
            metadatas=[{"source": source} for _ in parts],
        )
        reindexed += 1

    return {
        "poisoned": len(poisoned_sources),
        "reindexed": reindexed,
        "deleted_only": deleted_only,
        "failed": failed,
    }

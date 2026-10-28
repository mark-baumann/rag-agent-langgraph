"""Lernmodus: PDF-Seiten anzeigen und per LLM verständlich erklären.

Jede Seite wird mit PyMuPDF als PNG gerendert (statt eines Base64-Iframes,
der in Streamlit browserabhängig nicht funktioniert, siehe AUG-280) und
zusammen mit ihrem Text an ein Vision-fähiges LLM gegeben. Das LLM fasst
zusammen, was auf der Seite steht, erklärt es in einfachen Worten, listet
Fachbegriffe und stellt Verständnisfragen.

Wie in answer_generation.py gilt: ohne OPENAI_API_KEY wird kein Fake-Text
erzeugt, sondern `None` zurückgegeben.
"""

from __future__ import annotations

import base64
import os

# Erklär-Niveaus, in der GUI wählbar.
LEVELS = {
    "einfach": "Erkläre so einfach, dass es ein 12-jähriges Kind versteht. "
    "Nutze Alltagsvergleiche und vermeide Fachsprache.",
    "normal": "Erkläre verständlich für einen Erwachsenen ohne Vorwissen im Fachgebiet. "
    "Fachbegriffe sind erlaubt, werden aber immer erklärt.",
    "fachlich": "Erkläre für jemanden mit Grundkenntnissen im Fachgebiet, der die "
    "Seite tiefer verstehen will: Zusammenhänge, Hintergründe, Konsequenzen.",
}

SYSTEM_PROMPT = (
    "Du bist ein geduldiger Nachhilfelehrer. Der Nutzer versteht ein Dokument "
    "nicht und schickt dir eine Seite daraus (als Bild und als extrahierten Text). "
    "Erkläre ihm, was auf der Seite steht, so dass er es wirklich versteht. "
    "Halte dich an den Inhalt der Seite und erfinde nichts dazu; wenn etwas "
    "unleserlich oder ohne Kontext unklar ist, sag das. Antworte auf Deutsch "
    "in Markdown."
)

EXPLAIN_TEMPLATE = (
    "Dokument: {doc_name} — Seite {page_no} von {total_pages}.\n\n"
    "Extrahierter Text der Seite:\n\"\"\"\n{page_text}\n\"\"\"\n\n"
    "Niveau: {level_hint}\n\n"
    "Gliedere deine Antwort genau so:\n"
    "### 📄 Was steht auf dieser Seite?\n"
    "Kurze Zusammenfassung in 2–4 Sätzen.\n\n"
    "### 💡 Erklärung\n"
    "Gehe die Seite Abschnitt für Abschnitt durch und erkläre jeden Teil "
    "verständlich. Erkläre auch Tabellen, Zahlen, Formeln und Grafiken.\n\n"
    "### 📚 Wichtige Begriffe\n"
    "Liste der Fachbegriffe/Abkürzungen auf der Seite mit kurzer Erklärung.\n\n"
    "### ✅ Teste dich selbst\n"
    "2–3 nummerierte Verständnisfragen.\n\n"
    "### 🔑 Lösungen\n"
    "Die Lösungen zu den Fragen, gleich nummeriert."
)

QUESTION_TEMPLATE = (
    "Dokument: {doc_name} — Seite {page_no}.\n\n"
    "Extrahierter Text der Seite:\n\"\"\"\n{page_text}\n\"\"\"\n\n"
    "Frage des Nutzers zu dieser Seite: {question}\n\n"
    "Beantworte die Frage verständlich und bezogen auf diese Seite. "
    "Niveau: {level_hint}"
)

# Sehr lange Seiten (z. B. Tabellen-Dumps) kürzen, damit der Prompt klein bleibt.
MAX_PAGE_CHARS = 12_000


def page_count(file_bytes: bytes) -> int:
    import fitz

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return doc.page_count


def render_page_png(file_bytes: bytes, page_index: int, zoom: float = 2.0) -> bytes:
    """Rendert eine Seite (0-basiert) als PNG-Bytes."""
    import fitz

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")


def extract_page_text(file_bytes: bytes, page_index: int) -> str:
    """Text einer einzelnen Seite; fällt für Scans auf Tesseract-OCR zurück."""
    import fitz

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        page = doc.load_page(page_index)
        text = page.get_text()
        if text.strip():
            return text
        try:
            tp = page.get_textpage_ocr(language="deu+eng", dpi=200, full=True)
            return tp.extractText()
        except Exception:
            return ""


def _user_content(prompt: str, page_png: bytes | None) -> list[dict] | str:
    if page_png is None:
        return prompt
    image_url = "data:image/png;base64," + base64.b64encode(page_png).decode("ascii")
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
    ]


def _chat(prompt: str, page_png: bytes | None, model: str) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(prompt, page_png)},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def build_explain_prompt(
    page_text: str,
    page_no: int,
    total_pages: int,
    doc_name: str,
    level: str = "normal",
) -> str:
    return EXPLAIN_TEMPLATE.format(
        doc_name=doc_name,
        page_no=page_no,
        total_pages=total_pages,
        page_text=page_text.strip()[:MAX_PAGE_CHARS] or "(kein Text extrahierbar — nutze das Bild)",
        level_hint=LEVELS.get(level, LEVELS["normal"]),
    )


def explain_page(
    page_text: str,
    page_no: int,
    total_pages: int,
    doc_name: str,
    level: str = "normal",
    page_png: bytes | None = None,
    model: str = "gpt-4o-mini",
) -> str | None:
    """Erklärt eine Seite (1-basierte ``page_no``). `None` ohne API-Key."""
    if not page_text.strip() and page_png is None:
        return None
    prompt = build_explain_prompt(page_text, page_no, total_pages, doc_name, level)
    return _chat(prompt, page_png, model)


def answer_page_question(
    question: str,
    page_text: str,
    page_no: int,
    doc_name: str,
    level: str = "normal",
    page_png: bytes | None = None,
    model: str = "gpt-4o-mini",
) -> str | None:
    """Beantwortet eine Rückfrage zu einer Seite. `None` ohne API-Key."""
    if not question.strip():
        return None
    prompt = QUESTION_TEMPLATE.format(
        doc_name=doc_name,
        page_no=page_no,
        page_text=page_text.strip()[:MAX_PAGE_CHARS] or "(kein Text extrahierbar — nutze das Bild)",
        question=question.strip(),
        level_hint=LEVELS.get(level, LEVELS["normal"]),
    )
    return _chat(prompt, page_png, model)

"""LLM-gestützte Antwortgenerierung aus Retrieval-Ergebnissen.

Die Retrieval-Schritte (vector_store.py) liefern nur Chunks + Ähnlichkeits-
Scores zurück — für eine tatsächliche Antwort auf die Nutzerfrage braucht es
zusätzlich einen Generation-Schritt (das "G" in RAG), der die Chunks einem
LLM als Kontext gibt und daraus eine Antwort formuliert.

Ohne konfigurierten API-Key (OPENAI_API_KEY) wird kein Fake-Text erzeugt,
sondern klar `None` zurückgegeben — der Aufrufer entscheidet, wie er das
anzeigt.
"""

from __future__ import annotations

import os

SYSTEM_PROMPT = (
    "Du beantwortest Fragen ausschließlich auf Basis der bereitgestellten "
    "Textausschnitte aus hochgeladenen PDF-Dokumenten. Wenn die Ausschnitte "
    "die Frage nicht beantworten, sag das klar und deutlich statt zu raten. "
    "Antworte präzise und auf Deutsch."
)


def build_prompt(query: str, chunks: list[str]) -> str:
    """Baut den User-Prompt aus Frage + Kontext-Chunks."""
    context = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))
    return (
        f"Kontext-Ausschnitte aus den Dokumenten:\n\n{context}\n\n"
        f"Frage: {query}\n\n"
        "Beantworte die Frage auf Basis der obigen Ausschnitte. "
        "Verweise dabei mit [1], [2], ... auf die verwendeten Ausschnitte."
    )


def generate_answer(
    query: str,
    chunks: list[str],
    model: str = "gpt-4o-mini",
) -> str | None:
    """Generiert eine Antwort aus Query + Retrieval-Chunks via LLM.

    Gibt `None` zurück, wenn kein API-Key konfiguriert ist oder kein Chunk
    vorliegt — der Aufrufer zeigt dann einen entsprechenden Hinweis statt
    einer generierten Antwort an.
    """
    if not chunks:
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, chunks)},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

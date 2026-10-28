"""Lernmodus: Aus einer Seite oder einem ganzen Dokument einen Podcast machen.

Zwei Schritte:
1. Ein LLM schreibt ein Gespräch zwischen zwei Sprechern — Lena stellt die
   Fragen, die sich ein Laie stellt, Max erklärt mit Beispielen.
2. Jede Gesprächszeile wird per OpenAI-TTS mit der Stimme des jeweiligen
   Sprechers vertont; die MP3-Stücke werden aneinandergehängt (MP3-Frames
   lassen sich direkt konkatenieren).

Wie in answer_generation.py gilt: ohne OPENAI_API_KEY → `None`.
"""

from __future__ import annotations

import json
import os

SPEAKERS = {
    "Lena": {
        "voice": "nova",
        "role": "Moderatorin, neugierig, stellt die Fragen, die sich ein Laie stellt, "
        "fasst zusammen und hakt nach, wenn etwas kompliziert klingt",
    },
    "Max": {
        "voice": "onyx",
        "role": "Experte, erklärt geduldig mit Alltagsbeispielen und Vergleichen",
    },
}

LENGTHS = {
    "kurz": "ca. 12–16 Wortwechsel (rund 3 Minuten)",
    "lang": "ca. 30–40 Wortwechsel (rund 8 Minuten)",
}

TTS_MODEL = "gpt-4o-mini-tts"
TTS_INSTRUCTIONS = (
    "Sprich natürlich, lebendig und locker auf Deutsch, wie in einem beliebten "
    "Wissens-Podcast. Betone wichtige Begriffe, mach kleine Pausen."
)
# Limit der TTS-API pro Anfrage.
MAX_TTS_CHARS = 4000
MAX_SOURCE_CHARS = 40_000

SYSTEM_PROMPT = (
    "Du schreibst Skripte für einen deutschen Lern-Podcast, der schwierige "
    "Dokumente so erklärt, dass jeder sie versteht. Das Gespräch ist "
    "unterhaltsam, mitreißend und trotzdem inhaltlich korrekt: Halte dich an "
    "den Inhalt des Dokuments und erfinde keine Fakten dazu. Antworte "
    "ausschließlich mit JSON."
)

SCRIPT_TEMPLATE = (
    "Quelle: {doc_name} ({scope}).\n\n"
    "Inhalt:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Schreibe ein Podcast-Gespräch zwischen zwei Sprechern:\n"
    "{speakers}\n\n"
    "Länge: {length}.\n"
    "Niveau: {level_hint}\n\n"
    "Aufbau: lockerer Einstieg mit Aufhänger, warum das Thema wichtig ist → "
    "Inhalt Schritt für Schritt verständlich erklären (Fachbegriffe immer "
    "auflösen, Zahlen und Tabellen einordnen, Beispiele aus dem Alltag) → "
    "kurze Zusammenfassung der 3 wichtigsten Punkte zum Merken → Verabschiedung.\n"
    "Schreibe gesprochene Sprache, keine Aufzählungszeichen, kein Markdown.\n\n"
    'Format: {{"title": "...", "segments": [{{"speaker": "Lena", "text": "..."}}, ...]}}'
)


def build_script_prompt(
    text: str,
    doc_name: str,
    scope: str,
    length: str = "kurz",
    level_hint: str = "",
) -> str:
    speakers = "\n".join(f"- {name}: {info['role']}" for name, info in SPEAKERS.items())
    return SCRIPT_TEMPLATE.format(
        doc_name=doc_name,
        scope=scope,
        text=text.strip()[:MAX_SOURCE_CHARS] or "(kein Text extrahierbar)",
        speakers=speakers,
        length=LENGTHS.get(length, LENGTHS["kurz"]),
        level_hint=level_hint or "verständlich für Laien",
    )


def parse_script(raw: str) -> dict | None:
    """Liest das JSON-Skript des LLM; unbekannte Sprecher/leere Zeilen fliegen raus."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    segments = [
        {"speaker": seg["speaker"], "text": seg["text"].strip()}
        for seg in data.get("segments", [])
        if isinstance(seg, dict)
        and seg.get("speaker") in SPEAKERS
        and isinstance(seg.get("text"), str)
        and seg["text"].strip()
    ]
    if not segments:
        return None
    return {"title": str(data.get("title") or "Podcast"), "segments": segments}


def _client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def generate_podcast_script(
    text: str,
    doc_name: str,
    scope: str,
    length: str = "kurz",
    level_hint: str = "",
    model: str = "gpt-4o-mini",
) -> dict | None:
    """Schreibt das Podcast-Skript. `None` ohne API-Key, Text oder gültige Antwort."""
    if not text.strip():
        return None
    client = _client()
    if client is None:
        return None
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_script_prompt(text, doc_name, scope, length, level_hint),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
    )
    return parse_script(response.choices[0].message.content)


def synthesize_podcast(script: dict, on_progress=None) -> bytes | None:
    """Vertont das Skript Zeile für Zeile und liefert eine MP3-Datei."""
    client = _client()
    if client is None:
        return None
    segments = script["segments"]
    audio = bytearray()
    for i, seg in enumerate(segments):
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=SPEAKERS[seg["speaker"]]["voice"],
            input=seg["text"][:MAX_TTS_CHARS],
            instructions=TTS_INSTRUCTIONS,
            response_format="mp3",
        )
        audio.extend(response.read())
        if on_progress is not None:
            on_progress((i + 1) / len(segments))
    return bytes(audio)

"""Tests für den Podcast-Generator (podcast.py)."""

import json
from unittest.mock import MagicMock, patch

from podcast import (
    SPEAKERS,
    build_script_prompt,
    generate_podcast_script,
    parse_script,
    synthesize_podcast,
)

SCRIPT = {
    "title": "Photosynthese einfach erklärt",
    "segments": [
        {"speaker": "Lena", "text": "Max, was ist Photosynthese?"},
        {"speaker": "Max", "text": "Pflanzen machen aus Licht Zucker."},
    ],
}


class TestBuildScriptPrompt:
    def test_contains_source_speakers_and_length(self):
        prompt = build_script_prompt("Chlorophyll", "bio.pdf", "Seite 2 von 5", length="lang")
        assert "Chlorophyll" in prompt
        assert "bio.pdf (Seite 2 von 5)" in prompt
        assert "Lena" in prompt and "Max" in prompt
        assert "8 Minuten" in prompt


class TestParseScript:
    def test_valid_script(self):
        assert parse_script(json.dumps(SCRIPT)) == SCRIPT

    def test_drops_unknown_speakers_and_empty_lines(self):
        raw = json.dumps(
            {
                "segments": [
                    {"speaker": "Lena", "text": " Hallo "},
                    {"speaker": "Bob", "text": "Wer bin ich?"},
                    {"speaker": "Max", "text": "   "},
                ]
            }
        )
        assert parse_script(raw) == {
            "title": "Podcast",
            "segments": [{"speaker": "Lena", "text": "Hallo"}],
        }

    def test_invalid_json_returns_none(self):
        assert parse_script("kein json") is None
        assert parse_script(json.dumps({"segments": []})) is None


class TestGeneratePodcastScript:
    def test_missing_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert generate_podcast_script("Text", "doc.pdf", "Seite 1") is None

    def test_empty_text_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert generate_podcast_script("  ", "doc.pdf", "Seite 1") is None

    def test_requests_json_and_parses(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=json.dumps(SCRIPT)))]
        client = MagicMock()
        client.chat.completions.create.return_value = response

        with patch("openai.OpenAI", return_value=client):
            result = generate_podcast_script("Text", "doc.pdf", "Seite 1")

        assert result == SCRIPT
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["response_format"] == {"type": "json_object"}


class TestSynthesizePodcast:
    def test_missing_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert synthesize_podcast(SCRIPT) is None

    def test_one_tts_call_per_segment_with_speaker_voice(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = MagicMock()
        client.audio.speech.create.side_effect = [
            MagicMock(read=MagicMock(return_value=b"AAA")),
            MagicMock(read=MagicMock(return_value=b"BBB")),
        ]
        progress = []

        with patch("openai.OpenAI", return_value=client):
            audio = synthesize_podcast(SCRIPT, on_progress=progress.append)

        assert audio == b"AAABBB"
        assert progress == [0.5, 1.0]
        voices = [c.kwargs["voice"] for c in client.audio.speech.create.call_args_list]
        assert voices == [SPEAKERS["Lena"]["voice"], SPEAKERS["Max"]["voice"]]

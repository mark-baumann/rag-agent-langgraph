"""Tests für den Lernmodus: Seiten rendern und per LLM erklären (page_explainer.py)."""

from unittest.mock import MagicMock, patch

import fitz
import pytest

from page_explainer import (
    answer_page_question,
    build_explain_prompt,
    explain_page,
    extract_page_text,
    page_count,
    render_page_png,
)


@pytest.fixture
def pdf_bytes():
    doc = fitz.open()
    for text in ("Erste Seite: Photosynthese", "Zweite Seite: Chlorophyll"):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _mock_client(content):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


class TestPdfPages:
    def test_page_count(self, pdf_bytes):
        assert page_count(pdf_bytes) == 2

    def test_render_page_png(self, pdf_bytes):
        png = render_page_png(pdf_bytes, 1)
        assert png.startswith(b"\x89PNG")

    def test_extract_page_text_is_per_page(self, pdf_bytes):
        assert "Photosynthese" in extract_page_text(pdf_bytes, 0)
        second = extract_page_text(pdf_bytes, 1)
        assert "Chlorophyll" in second
        assert "Photosynthese" not in second


class TestBuildExplainPrompt:
    def test_contains_page_info_and_sections(self):
        prompt = build_explain_prompt("Text X", 3, 10, "doc.pdf", level="einfach")
        assert "Seite 3 von 10" in prompt
        assert "doc.pdf" in prompt
        assert "Text X" in prompt
        assert "12-jähriges Kind" in prompt
        assert "Wichtige Begriffe" in prompt

    def test_empty_text_points_to_image(self):
        prompt = build_explain_prompt("   ", 1, 1, "scan.pdf")
        assert "nutze das Bild" in prompt


class TestExplainPage:
    def test_missing_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert explain_page("Text", 1, 1, "doc.pdf") is None

    def test_nothing_to_explain_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert explain_page("", 1, 1, "doc.pdf") is None

    def test_sends_text_and_image(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = _mock_client("Erklärung")

        with patch("openai.OpenAI", return_value=client):
            result = explain_page("Seitentext", 1, 2, "doc.pdf", page_png=b"\x89PNGfake")

        assert result == "Erklärung"
        _, kwargs = client.chat.completions.create.call_args
        user_content = kwargs["messages"][1]["content"]
        assert user_content[0]["type"] == "text"
        assert "Seitentext" in user_content[0]["text"]
        assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_text_only_without_image(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = _mock_client("Erklärung")

        with patch("openai.OpenAI", return_value=client):
            explain_page("Seitentext", 1, 1, "doc.pdf")

        _, kwargs = client.chat.completions.create.call_args
        assert isinstance(kwargs["messages"][1]["content"], str)


class TestAnswerPageQuestion:
    def test_empty_question_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert answer_page_question("  ", "Text", 1, "doc.pdf") is None

    def test_question_in_prompt(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = _mock_client("Antwort")

        with patch("openai.OpenAI", return_value=client):
            result = answer_page_question("Was heißt das?", "Seitentext", 4, "doc.pdf")

        assert result == "Antwort"
        _, kwargs = client.chat.completions.create.call_args
        assert "Was heißt das?" in kwargs["messages"][1]["content"]
        assert "Seite 4" in kwargs["messages"][1]["content"]

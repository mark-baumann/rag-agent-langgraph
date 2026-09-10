"""Tests für die LLM-gestützte Antwortgenerierung (answer_generation.py)."""

from unittest.mock import MagicMock, patch

from answer_generation import build_prompt, generate_answer


class TestBuildPrompt:
    def test_includes_query_and_chunks(self):
        prompt = build_prompt("Was ist X?", ["Chunk A", "Chunk B"])
        assert "Was ist X?" in prompt
        assert "[1] Chunk A" in prompt
        assert "[2] Chunk B" in prompt


class TestGenerateAnswer:
    def test_no_chunks_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert generate_answer("Frage?", []) is None

    def test_missing_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert generate_answer("Frage?", ["Ein Chunk"]) is None

    def test_calls_llm_and_returns_content(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Die Antwort lautet 42."))]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client) as mock_openai:
            result = generate_answer("Was ist die Antwort?", ["Kontext-Chunk"])

        mock_openai.assert_called_once_with(api_key="sk-test")
        assert result == "Die Antwort lautet 42."

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4o-mini"
        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "Was ist die Antwort?" in messages[1]["content"]

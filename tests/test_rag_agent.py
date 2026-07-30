"""
Tests für den RAG Agent mit LangGraph.

Testet:
- SimpleVectorStore: add, search, cosine similarity
- SimpleEmbedder: fit, embed, TF-IDF
- RAGPipeline: retrieve, generate, query
- RAGState: TypedDict-Struktur
"""

import numpy as np
import pytest

from rag_agent import (
    RAGPipeline,
    RAGState,
    SimpleEmbedder,
    SimpleVectorStore,
)

# ═══════════════════════════════════════════════════════════════
# RAGState Tests
# ═══════════════════════════════════════════════════════════════

class TestRAGState:
    """Testet die RAGState TypedDict-Struktur."""

    def test_ragstate_creation(self):
        """RAGState kann mit allen Feldern erstellt werden."""
        state: RAGState = {
            "query": "Testfrage",
            "retrieved_docs": ["Dok 1", "Dok 2"],
            "context": "Kontext",
            "answer": "Antwort",
            "needs_rewrite": False,
            "iteration": 0,
        }
        assert state["query"] == "Testfrage"
        assert len(state["retrieved_docs"]) == 2
        assert state["needs_rewrite"] is False
        assert state["iteration"] == 0

    def test_ragstate_annotated_add(self):
        """retrieved_docs nutzt operator.add für Annotation."""
        state1: RAGState = {
            "query": "q",
            "retrieved_docs": ["A"],
            "context": "",
            "answer": "",
            "needs_rewrite": False,
            "iteration": 0,
        }
        state2: RAGState = {
            "query": "q",
            "retrieved_docs": ["B"],
            "context": "",
            "answer": "",
            "needs_rewrite": False,
            "iteration": 0,
        }
        # operator.add konkateniert Listen
        combined = state1["retrieved_docs"] + state2["retrieved_docs"]
        assert combined == ["A", "B"]


# ═══════════════════════════════════════════════════════════════
# SimpleVectorStore Tests
# ═══════════════════════════════════════════════════════════════

class TestSimpleVectorStore:
    """Testet den SimpleVectorStore."""

    def test_empty_store_returns_empty(self):
        """Leerer Store gibt leere Liste zurück."""
        store = SimpleVectorStore()
        result = store.search(np.array([1.0, 0.0]))
        assert result == []

    def test_add_and_search_single_doc(self):
        """Ein Dokument hinzufügen und suchen."""
        store = SimpleVectorStore()
        store.add(["Dokument 1"], [np.array([1.0, 0.0, 0.0])])
        result = store.search(np.array([1.0, 0.0, 0.0]), k=1)
        assert result == ["Dokument 1"]

    def test_search_returns_top_k(self):
        """Suche gibt die k ähnlichsten Dokumente zurück."""
        store = SimpleVectorStore()
        store.add(
            ["Dok A", "Dok B", "Dok C"],
            [
                np.array([1.0, 0.0, 0.0]),  # ähnlich zu query
                np.array([0.0, 1.0, 0.0]),  # orthogonal
                np.array([0.5, 0.0, 0.0]),  # teilweise ähnlich
            ],
        )
        # Query nahe an Dok A
        result = store.search(np.array([1.0, 0.1, 0.0]), k=2)
        assert len(result) == 2
        assert "Dok A" in result

    def test_cosine_similarity_perfect_match(self):
        """Perfekte Übereinstimmung (gleicher Vektor) gibt Score ~1.0."""
        store = SimpleVectorStore()
        vec = np.array([0.6, 0.8])
        store.add(["Perfekt"], [vec])
        result = store.search(vec, k=1)
        assert result == ["Perfekt"]

    def test_add_multiple_batches(self):
        """Mehrere Batches hinzufügen."""
        store = SimpleVectorStore()
        store.add(["A"], [np.array([1.0, 0.0])])
        store.add(["B"], [np.array([0.0, 1.0])])
        assert len(store.documents) == 2
        assert len(store.embeddings) == 2


# ═══════════════════════════════════════════════════════════════
# SimpleEmbedder Tests
# ═══════════════════════════════════════════════════════════════

class TestSimpleEmbedder:
    """Testet den SimpleEmbedder."""

    def test_fit_builds_vocabulary(self):
        """fit() baut Vokabular aus Dokumenten auf."""
        embedder = SimpleEmbedder()
        docs = ["Hallo Welt", "Welt der KI"]
        embedder.fit(docs)
        assert len(embedder.vocab) > 0
        assert "hallo" in embedder.vocab
        assert "welt" in embedder.vocab

    def test_embed_returns_vector(self):
        """embed() gibt einen numpy-Vektor zurück."""
        embedder = SimpleEmbedder()
        embedder.fit(["Hallo Welt", "KI Agent"])
        vec = embedder.embed("Hallo KI")
        assert isinstance(vec, np.ndarray)
        assert len(vec) == len(embedder.vocab)

    def test_embed_unknown_tokens(self):
        """Unbekannte Tokens werden ignoriert (kein Crash)."""
        embedder = SimpleEmbedder()
        embedder.fit(["Hallo Welt"])
        vec = embedder.embed("Unbekanntes Token")
        assert np.all(vec == 0.0)  # Alles Null für unbekannte Tokens

    def test_embed_docs_with_shared_tokens_higher_similarity(self):
        """Dokumente mit gemeinsamen Tokens haben höhere Ähnlichkeit."""
        embedder = SimpleEmbedder()
        docs = [
            "Python Machine Learning Basics",
            "Python Deep Learning Advanced",
            "Fußball Bundesliga Ergebnisse",
        ]
        embedder.fit(docs)

        # "Python ML" teilt "python" mit doc1 und doc2
        vec_python_ml = embedder.embed("Python Machine")
        # "Python DL" teilt "python" mit doc1 und doc2
        vec_python_dl = embedder.embed("Python Deep")
        # Kein shared token mit doc1/doc2
        vec_fussball = embedder.embed("Fußball Bundesliga")

        sim_python = float(np.dot(vec_python_ml, vec_python_dl))
        sim_ml_fb = float(np.dot(vec_python_ml, vec_fussball))

        # Python-basierte Queries sollten ähnlicher sein als mit Fußball
        assert sim_python > sim_ml_fb, (
            f"Python-Ähnlichkeit={sim_python:.4f} sollte > ML-FB={sim_ml_fb:.4f}"
        )


# ═══════════════════════════════════════════════════════════════
# RAGPipeline Tests
# ═══════════════════════════════════════════════════════════════

class TestRAGPipeline:
    """Testet die RAGPipeline."""

    @pytest.fixture
    def documents(self):
        return [
            "Vergütungsvereinbarung 2025: Basisfallwert 4.200 Euro.",
            "Vergütungsvereinbarung 2026: Basisfallwert 4.350 Euro.",
            "SGB V §87: Vergütung nach Fallpauschalen.",
            "DRG-System wurde 2003 eingeführt.",
            "Qualitätsberichte müssen jährlich veröffentlicht werden.",
            "Pflegepersonaluntergrenzen seit 2019 verbindlich.",
            "Hybrid-DRG: Neue Vergütungsform ab 2024.",
            "Notfallversorgung: Reform in Planung.",
        ]

    @pytest.fixture
    def pipeline(self, documents):
        return RAGPipeline(documents)

    def test_retrieve_returns_docs(self, pipeline):
        """retrieve() gibt Dokumente zurück."""
        docs = pipeline.retrieve("Basisfallwert 2025", k=3)
        assert len(docs) == 3
        assert any("Basisfallwert" in doc for doc in docs)

    def test_retrieve_respects_k(self, pipeline):
        """retrieve() respektiert den k-Parameter."""
        docs = pipeline.retrieve("Vergütung", k=2)
        assert len(docs) == 2

    def test_generate_with_docs(self, pipeline):
        """generate() mit Dokumenten produziert Antwort."""
        docs = ["Dok A", "Dok B"]
        answer = pipeline.generate("Testfrage", docs)
        assert "2 Dokumenten" in answer
        assert "Dok A" in answer
        assert "Testfrage" in answer

    def test_generate_empty_docs(self, pipeline):
        """generate() ohne Dokumente gibt Hinweis."""
        answer = pipeline.generate("Frage", [])
        assert "Keine relevanten Dokumente" in answer

    def test_query_returns_dict(self, pipeline):
        """query() gibt ein Dict mit allen Feldern zurück."""
        result = pipeline.query("Basisfallwert")
        assert isinstance(result, dict)
        assert "query" in result
        assert "retrieved_docs" in result
        assert "answer" in result
        assert result["query"] == "Basisfallwert"

    def test_query_relevant_results(self, pipeline):
        """query() findet relevante Ergebnisse für bekannte Themen."""
        result = pipeline.query("DRG-System", k=3)
        assert any("DRG" in doc for doc in result["retrieved_docs"])

    def test_pipeline_initialization(self, documents):
        """Pipeline-Initialisierung speichert Dokumente."""
        pipeline = RAGPipeline(documents)
        assert len(pipeline.store.documents) == len(documents)
        assert len(pipeline.store.embeddings) == len(documents)
        assert len(pipeline.embedder.vocab) > 0

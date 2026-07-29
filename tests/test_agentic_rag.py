"""
Tests für Agentic RAG Modul.
"""
import pytest
import numpy as np
from agentic_rag import (
    AgenticRAG,
    AgenticRAGState,
    RetrievalStrategy,
    EvidenceVerifier,
    QueryDecomposer,
)


class MockEmbedder:
    def embed(self, text: str) -> np.ndarray:
        return np.random.randn(128)


class MockVectorStore:
    def __init__(self):
        self.docs = [
            "Reinforcement Learning ist ein Teilgebiet des Machine Learning.",
            "RAG kombiniert Retrieval mit Generation für bessere Antworten.",
            "Agentic RAG verwendet Multi-Step Reasoning.",
            "LangGraph ist ein Framework für Stateful Agents.",
            "Python ist eine Programmiersprache.",
        ]

    def search(self, query_embedding, k=3):
        return self.docs[:k]


class MockLLM:
    def generate(self, prompt: str) -> str:
        return f"Generierte Antwort basierend auf Kontext."


class TestQueryDecomposer:
    def test_simple_query(self):
        d = QueryDecomposer()
        result = d.decompose("Was ist RAG?")
        assert "Was ist RAG?" in result

    def test_comparison_query(self):
        d = QueryDecomposer()
        result = d.decompose("RAG vs Agentic RAG")
        assert len(result) >= 2
        assert any("RAG" in r for r in result)
        assert any("Agentic RAG" in r for r in result)

    def test_and_query(self):
        d = QueryDecomposer()
        result = d.decompose("RAG und LangGraph und Python")
        assert len(result) >= 3


class TestRetrievalStrategy:
    def test_fact_query_selects_sparse(self):
        s = RetrievalStrategy()
        assert s.select("Wann wurde RAG erfunden?", []) == "sparse"

    def test_conceptual_query_selects_dense(self):
        s = RetrievalStrategy()
        assert s.select("Wie funktioniert RAG?", []) == "dense"

    def test_few_results_selects_dense(self):
        s = RetrievalStrategy()
        assert s.select("Was ist RAG?", [{"content": "x"}]) == "dense"


class TestEvidenceVerifier:
    def test_verified_answer(self):
        v = EvidenceVerifier()
        docs = [{"content": "RAG kombiniert Retrieval mit Generation.", "source": "doc1"}]
        result = v.verify("RAG kombiniert Retrieval mit Generation.", docs)
        assert result["verified"] is True
        assert "doc1" in result["citations"]

    def test_unverified_answer(self):
        v = EvidenceVerifier()
        docs = [{"content": "RAG ist ein Framework.", "source": "doc1"}]
        result = v.verify("RAG wurde 2020 von Google erfunden und ist das beste System.", docs)
        assert result["verified"] is False
        assert len(result["unsupported_claims"]) > 0

    def test_claim_extraction(self):
        v = EvidenceVerifier()
        claims = v._extract_claims("Erster Satz. Zweiter Satz! Dritter Satz?")
        assert len(claims) == 3


class TestAgenticRAG:
    def test_full_workflow(self):
        rag = AgenticRAG(
            vector_store=MockVectorStore(),
            embedder=MockEmbedder(),
            llm=MockLLM(),
            max_iterations=2,
        )
        result = rag.run("Was ist RAG?")
        assert "answer" in result
        assert "citations" in result
        assert "confidence" in result
        assert result["iterations"] <= 2
        assert result["docs_retrieved"] > 0

    def test_max_iterations_respected(self):
        rag = AgenticRAG(
            vector_store=MockVectorStore(),
            embedder=MockEmbedder(),
            llm=MockLLM(),
            max_iterations=1,
        )
        result = rag.run("Komplexe Query mit vs Vergleich und mehreren Aspekten")
        assert result["iterations"] <= 1

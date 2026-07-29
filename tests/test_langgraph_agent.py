"""
Tests für den LangGraph RAG Agent.

Testet:
- RAGAgent: retrieve, grade_docs, generate, check_hallucination, rewrite
- Graph-Bau: build_graph(), run()
- Conditional Edges: should_rewrite, should_generate
- Tools: calculator, web_search_stub
- AgentState: TypedDict-Struktur
"""

import numpy as np
import pytest

from langgraph_agent import (
    AgentState,
    RAGAgent,
    calculator,
    web_search_stub,
    TOOLS,
    HAS_LANGGRAPH,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def documents():
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
def agent(documents):
    return RAGAgent(documents, max_iterations=3)


@pytest.fixture
def initial_state():
    return AgentState(
        query="Testfrage",
        retrieved_docs=[],
        graded_docs=[],
        answer="",
        needs_rewrite=False,
        hallucination_score=0.0,
        iteration=0,
        tool_calls=[],
    )


# ═══════════════════════════════════════════════════════════════
# AgentState Tests
# ═══════════════════════════════════════════════════════════════

class TestAgentState:
    """Testet die AgentState TypedDict-Struktur."""

    def test_agentstate_creation(self):
        """AgentState kann mit allen Feldern erstellt werden."""
        state: AgentState = {
            "query": "Frage",
            "retrieved_docs": ["Dok 1"],
            "graded_docs": ["Dok 1"],
            "answer": "Antwort",
            "needs_rewrite": False,
            "hallucination_score": 0.1,
            "iteration": 1,
            "tool_calls": ["calculator"],
        }
        assert state["query"] == "Frage"
        assert state["iteration"] == 1
        assert state["hallucination_score"] == 0.1

    def test_agentstate_annotated_add(self):
        """retrieved_docs nutzt operator.add für Annotation."""
        state1: AgentState = {
            "query": "q", "retrieved_docs": ["A"], "graded_docs": [],
            "answer": "", "needs_rewrite": False,
            "hallucination_score": 0.0, "iteration": 0, "tool_calls": [],
        }
        state2: AgentState = {
            "query": "q", "retrieved_docs": ["B"], "graded_docs": [],
            "answer": "", "needs_rewrite": False,
            "hallucination_score": 0.0, "iteration": 0, "tool_calls": [],
        }
        combined = state1["retrieved_docs"] + state2["retrieved_docs"]
        assert combined == ["A", "B"]


# ═══════════════════════════════════════════════════════════════
# Tools Tests
# ═══════════════════════════════════════════════════════════════

class TestTools:
    """Testet die Tool-Funktionen."""

    def test_calculator_basic(self):
        """calculator() wertet einfache Ausdrücke aus."""
        result = calculator("2 + 2")
        assert "4" in result

    def test_calculator_math_functions(self):
        """calculator() unterstützt math-Funktionen."""
        result = calculator("sqrt(16)")
        assert "4.0" in result

    def test_calculator_error(self):
        """calculator() fängt Fehler ab."""
        result = calculator("1/0")
        assert "Fehler" in result

    def test_web_search_stub(self):
        """web_search_stub() gibt Stub-Antwort zurück."""
        result = web_search_stub("Python LangGraph")
        assert "Python LangGraph" in result
        assert "Stub" in result

    def test_tools_dict(self):
        """TOOLS dict enthält calculator und web_search."""
        assert "calculator" in TOOLS
        assert "web_search" in TOOLS
        assert callable(TOOLS["calculator"])
        assert callable(TOOLS["web_search"])


# ═══════════════════════════════════════════════════════════════
# RAGAgent Node Tests
# ═══════════════════════════════════════════════════════════════

class TestRAGAgentNodes:
    """Testet die einzelnen Nodes des RAGAgent."""

    def test_retrieve_returns_docs(self, agent, initial_state):
        """retrieve() findet Dokumente."""
        state = initial_state.copy()
        state["query"] = "Basisfallwert 2025"
        result = agent.retrieve(state)
        assert len(result["retrieved_docs"]) == 3
        assert any("Basisfallwert" in doc for doc in result["retrieved_docs"])

    def test_grade_docs_filters_relevant(self, agent):
        """grade_docs() filtert relevante Dokumente."""
        state: AgentState = {
            "query": "Basisfallwert",
            "retrieved_docs": [
                "Vergütungsvereinbarung 2025: Basisfallwert 4.200 Euro.",
                "Notfallversorgung: Reform in Planung.",
                "Qualitätsberichte müssen jährlich veröffentlicht werden.",
            ],
            "graded_docs": [],
            "answer": "",
            "needs_rewrite": False,
            "hallucination_score": 0.0,
            "iteration": 0,
            "tool_calls": [],
        }
        result = agent.grade_docs(state)
        assert len(result["graded_docs"]) >= 1
        assert any("Basisfallwert" in doc for doc in result["graded_docs"])

    def test_grade_docs_fallback(self, agent):
        """grade_docs() verwendet Fallback wenn keine Docs matchen."""
        state: AgentState = {
            "query": "xyz_unbekannt_123",
            "retrieved_docs": [
                "Vergütungsvereinbarung 2025: Basisfallwert 4.200 Euro.",
                "Notfallversorgung: Reform in Planung.",
            ],
            "graded_docs": [],
            "answer": "",
            "needs_rewrite": False,
            "hallucination_score": 0.0,
            "iteration": 0,
            "tool_calls": [],
        }
        result = agent.grade_docs(state)
        # Fallback: alle retrieved_docs werden verwendet
        assert len(result["graded_docs"]) == 2

    def test_generate_produces_answer(self, agent):
        """generate() erzeugt Antwort aus Dokumenten."""
        state: AgentState = {
            "query": "Testfrage",
            "retrieved_docs": ["Dok A", "Dok B"],
            "graded_docs": ["Dok A", "Dok B"],
            "answer": "",
            "needs_rewrite": False,
            "hallucination_score": 0.0,
            "iteration": 0,
            "tool_calls": [],
        }
        result = agent.generate(state)
        assert "2 Dokumenten" in result["answer"]
        assert "Dok A" in result["answer"]
        assert "Testfrage" in result["answer"]

    def test_generate_empty_docs(self, agent):
        """generate() ohne Dokumente gibt Hinweis."""
        state: AgentState = {
            "query": "Frage",
            "retrieved_docs": [],
            "graded_docs": [],
            "answer": "",
            "needs_rewrite": False,
            "hallucination_score": 0.0,
            "iteration": 0,
            "tool_calls": [],
        }
        result = agent.generate(state)
        assert "Keine relevanten Dokumente" in result["answer"]

    def test_check_hallucination_low_score(self, agent):
        """check_hallucination() erkennt gute Antworten (niedriger Score)."""
        state: AgentState = {
            "query": "Basisfallwert",
            "retrieved_docs": ["Basisfallwert 2025: 4.200 Euro"],
            "graded_docs": ["Basisfallwert 2025: 4.200 Euro"],
            "answer": "Der Basisfallwert 2025 beträgt 4.200 Euro",
            "needs_rewrite": False,
            "hallucination_score": 0.0,
            "iteration": 0,
            "tool_calls": [],
        }
        result = agent.check_hallucination(state)
        assert result["needs_rewrite"] is False
        assert result["hallucination_score"] <= 0.5

    def test_check_hallucination_high_score(self, agent):
        """check_hallucination() erkennt Halluzination (hoher Score)."""
        state: AgentState = {
            "query": "Basisfallwert",
            "retrieved_docs": ["Basisfallwert 2025: 4.200 Euro"],
            "graded_docs": ["Basisfallwert 2025: 4.200 Euro"],
            "answer": "Der Mars ist 225 Millionen km von der Erde entfernt",
            "needs_rewrite": False,
            "hallucination_score": 0.0,
            "iteration": 0,
            "tool_calls": [],
        }
        result = agent.check_hallucination(state)
        # Hoher Score = viele Wörter nicht in Docs
        assert result["hallucination_score"] > 0.5

    def test_rewrite_increments_iteration(self, agent):
        """rewrite() erhöht den Iterationszähler."""
        state: AgentState = {
            "query": "Testfrage",
            "retrieved_docs": [],
            "graded_docs": [],
            "answer": "",
            "needs_rewrite": True,
            "hallucination_score": 0.0,
            "iteration": 1,
            "tool_calls": [],
        }
        result = agent.rewrite(state)
        assert result["iteration"] == 2
        assert "reformuliert" in result["query"]


# ═══════════════════════════════════════════════════════════════
# Router Tests
# ═══════════════════════════════════════════════════════════════

class TestRouters:
    """Testet die Conditional-Edge-Router."""

    def test_should_rewrite_true(self, agent):
        """should_rewrite() → 'rewrite' wenn needs_rewrite=True."""
        state: AgentState = {
            "query": "q", "retrieved_docs": [], "graded_docs": [],
            "answer": "", "needs_rewrite": True,
            "hallucination_score": 0.0, "iteration": 0, "tool_calls": [],
        }
        assert agent.should_rewrite(state) == "rewrite"

    def test_should_rewrite_false(self, agent):
        """should_rewrite() → 'end' wenn needs_rewrite=False."""
        state: AgentState = {
            "query": "q", "retrieved_docs": [], "graded_docs": [],
            "answer": "", "needs_rewrite": False,
            "hallucination_score": 0.0, "iteration": 0, "tool_calls": [],
        }
        assert agent.should_rewrite(state) == "end"

    def test_should_generate_true(self, agent):
        """should_generate() → 'generate' wenn Docs vorhanden."""
        state: AgentState = {
            "query": "q", "retrieved_docs": [], "graded_docs": ["Dok A"],
            "answer": "", "needs_rewrite": False,
            "hallucination_score": 0.0, "iteration": 0, "tool_calls": [],
        }
        assert agent.should_generate(state) == "generate"

    def test_should_generate_false(self, agent):
        """should_generate() → 'end' wenn keine Docs."""
        state: AgentState = {
            "query": "q", "retrieved_docs": [], "graded_docs": [],
            "answer": "", "needs_rewrite": False,
            "hallucination_score": 0.0, "iteration": 0, "tool_calls": [],
        }
        assert agent.should_generate(state) == "end"


# ═══════════════════════════════════════════════════════════════
# Graph & Integration Tests
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_LANGGRAPH, reason="LangGraph nicht installiert")
class TestGraphIntegration:
    """Integrationstests für den kompletten LangGraph-Workflow."""

    def test_build_graph_returns_compiled_graph(self, agent):
        """build_graph() gibt kompilierten Graphen zurück."""
        graph = agent.build_graph()
        assert graph is not None
        # Graph sollte invoke-Methode haben
        assert hasattr(graph, "invoke")

    def test_run_returns_agentstate(self, agent):
        """run() gibt einen vollständigen AgentState zurück."""
        result = agent.run("Basisfallwert 2025")
        assert isinstance(result, dict)
        assert "query" in result
        assert "retrieved_docs" in result
        assert "answer" in result
        assert "iteration" in result
        assert "hallucination_score" in result

    def test_run_finds_relevant_docs(self, agent):
        """run() findet relevante Dokumente für bekannte Query."""
        result = agent.run("Basisfallwert 2025")
        assert len(result["retrieved_docs"]) > 0
        assert any("Basisfallwert" in doc for doc in result["retrieved_docs"])

    def test_run_produces_answer(self, agent):
        """run() produziert eine nicht-leere Antwort."""
        result = agent.run("Was ist das DRG-System?")
        assert len(result["answer"]) > 0
        assert "Dokumenten" in result["answer"]

    def test_run_no_hallucination_on_relevant_query(self, agent):
        """run() erkennt keine Halluzination bei relevanter Query."""
        result = agent.run("Vergütungsvereinbarung 2025")
        # Sollte nicht als Halluzination markiert werden
        assert result["needs_rewrite"] is False

    def test_run_iteration_starts_at_zero(self, agent):
        """run() startet mit iteration=0."""
        result = agent.run("Test")
        assert result["iteration"] == 0

    def test_graph_has_all_nodes(self, agent):
        """Graph enthält alle erwarteten Nodes."""
        graph = agent.build_graph()
        # Nodes sind im Graph registriert
        nodes = graph.get_graph().nodes
        node_names = {n for n in nodes}
        expected = {"retrieve", "grade_docs", "generate", "check_hallucination", "rewrite"}
        assert expected.issubset(node_names), f"Fehlende Nodes: {expected - node_names}"

    def test_graph_has_conditional_edges(self, agent):
        """Graph enthält Conditional Edges."""
        graph = agent.build_graph()
        edges = graph.get_graph().edges
        # Mindestens eine Conditional Edge
        assert len(edges) > 0

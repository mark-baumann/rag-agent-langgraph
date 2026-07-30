"""
LangGraph RAG Agent — State Machine für Retrieval-Augmented Generation
=====================================================================

Echter LangGraph-Workflow mit:
- StateGraph mit typisiertem State
- Retrieve → Grade → Generate → Hallucination Check
- Conditional Edges für Rewrite-Loop
- Tool-Use (Calculator, Web Search Stub)

Architektur:
    [Query] → retrieve → grade_docs → generate → check_hallucination
                  ↑                                      │
                  └──────── rewrite (bei Halluzination) ─┘

Verwendet LangGraph für den State-Flow.
"""

import math
import operator
from typing import Annotated, Literal, TypedDict

import numpy as np

from vector_store import SimpleEmbedder, SimpleVectorStore

# LangGraph — falls nicht installiert, wird in der Demo darauf hingewiesen
try:
    from langgraph.graph import END, StateGraph
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    StateGraph = None  # type: ignore
    END = None  # type: ignore


# ═══════════════════════════════════════════════════════════════
# State-Definition
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """State für den LangGraph RAG Agenten."""
    query: str
    retrieved_docs: Annotated[list[str], operator.add]
    graded_docs: list[str]
    answer: str
    needs_rewrite: bool
    hallucination_score: float
    iteration: int
    tool_calls: list[str]


# ═══════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════

def calculator(expression: str) -> str:
    """Rechner-Tool: Wertet mathematische Ausdrücke aus."""
    try:
        # Sicher: nur math-Funktionen und Zahlen
        allowed = {"__builtins__": None}, {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "log": math.log,
            "pi": math.pi, "e": math.e,
        }
        result = eval(expression, *allowed)
        return f"Rechner: {expression} = {result}"
    except Exception as e:
        return f"Rechner-Fehler: {e}"


def web_search_stub(query: str) -> str:
    """Web-Suche-Stub: Simuliert eine Websuche (in Produktion: Tavily/SerpAPI)."""
    return f"[Web-Suche für: '{query}'] Keine Live-API — Stub-Antwort."


TOOLS = {
    "calculator": calculator,
    "web_search": web_search_stub,
}


# ═══════════════════════════════════════════════════════════════
# LangGraph Nodes
# ═══════════════════════════════════════════════════════════════

class RAGAgent:
    """
    LangGraph-basierter RAG Agent mit State Machine.

    Workflow:
        retrieve → grade_docs → [generate → check_hallucination]
                                    ↑              │
                                    └── rewrite ───┘ (bei Halluzination)
    """

    def __init__(self, documents: list[str], max_iterations: int = 3):
        self.store = SimpleVectorStore()
        self.embedder = SimpleEmbedder()
        self.max_iterations = max_iterations

        # Dokumente embedden und speichern
        self.embedder.fit(documents)
        embeddings = [self.embedder.embed(doc) for doc in documents]
        self.store.add(documents, embeddings)

    # ── Nodes ─────────────────────────────────────────────────

    def retrieve(self, state: AgentState) -> dict:
        """Retrieval-Node: Query → Embedding → Vector Search."""
        query_emb = self.embedder.embed(state["query"])
        docs = self.store.search(query_emb, k=3)
        return {"retrieved_docs": docs}

    def grade_docs(self, state: AgentState) -> dict:
        """Grade-Node: Bewertet Relevanz der gefundenen Dokumente."""
        query_lower = state["query"].lower()
        query_words = set(query_lower.split())
        graded = []
        for doc in state["retrieved_docs"]:
            # Einfache Relevanz-Heuristik: Wortüberlappung
            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                graded.append(doc)
        # Fallback: Wenn keine bewertet wurden, alle retrieved_docs verwenden
        if not graded:
            graded = list(state["retrieved_docs"])
        return {"graded_docs": graded}

    def generate(self, state: AgentState) -> dict:
        """Generate-Node: Erzeugt Antwort aus relevanten Dokumenten."""
        docs = state.get("graded_docs", state["retrieved_docs"])
        if not docs:
            return {"answer": "Keine relevanten Dokumente gefunden."}

        context = "\n".join(f"• {doc}" for doc in docs)
        answer = (
            f"Basierend auf {len(docs)} Dokumenten:\n\n"
            f"{context}\n\n"
            f"→ Antwort auf: \"{state['query']}\""
        )
        return {"answer": answer}

    def check_hallucination(self, state: AgentState) -> dict:
        """Hallucination-Check: Prüft ob Antwort auf Dokumenten basiert."""
        answer = state.get("answer", "")
        docs = state.get("graded_docs", state.get("retrieved_docs", []))

        if not docs or not answer:
            return {"needs_rewrite": False, "hallucination_score": 0.0}

        # Einfache Heuristik: Wie viele Wörter der Antwort kommen in den Docs vor?
        answer_words = set(answer.lower().split())
        doc_words = set()
        for doc in docs:
            doc_words.update(doc.lower().split())

        if not answer_words:
            return {"needs_rewrite": False, "hallucination_score": 0.0}

        overlap = len(answer_words & doc_words) / len(answer_words)
        needs_rewrite = overlap < 0.3 and state["iteration"] < self.max_iterations

        return {
            "needs_rewrite": needs_rewrite,
            "hallucination_score": round(1.0 - overlap, 3),
        }

    def rewrite(self, state: AgentState) -> dict:
        """Rewrite-Node: Formuliert Query um für besseres Retrieval."""
        query = state["query"]
        # Einfache Rewrite-Strategie: Keywords extrahieren
        keywords = [w for w in query.lower().split() if len(w) > 3]
        rewritten = f"{query} (reformuliert: {' '.join(keywords)})"
        return {
            "query": rewritten,
            "iteration": state["iteration"] + 1,
        }

    # ── Router ─────────────────────────────────────────────────

    def should_rewrite(self, state: AgentState) -> Literal["rewrite", "end"]:
        """Conditional Edge: Soll die Query umformuliert werden?"""
        if state.get("needs_rewrite", False):
            return "rewrite"
        return "end"

    def should_generate(self, state: AgentState) -> Literal["generate", "end"]:
        """Conditional Edge: Gibt es relevante Docs zum Generieren?"""
        if state.get("graded_docs"):
            return "generate"
        return "end"

    # ── Graph bauen ────────────────────────────────────────────

    def build_graph(self):
        """
        Baut den LangGraph StateGraph.

        Returns:
            Kompilierter Graph (Runnable).
        """
        if not HAS_LANGGRAPH:
            raise ImportError(
                "LangGraph ist nicht installiert. "
                "Installiere mit: uv pip install langgraph"
            )

        workflow = StateGraph(AgentState)

        # Nodes hinzufügen
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_docs", self.grade_docs)
        workflow.add_node("generate", self.generate)
        workflow.add_node("check_hallucination", self.check_hallucination)
        workflow.add_node("rewrite", self.rewrite)

        # Entry Point
        workflow.set_entry_point("retrieve")

        # Edges
        workflow.add_edge("retrieve", "grade_docs")

        # Conditional: grade_docs → generate oder END
        workflow.add_conditional_edges(
            "grade_docs",
            self.should_generate,
            {"generate": "generate", "end": END},
        )

        workflow.add_edge("generate", "check_hallucination")

        # Conditional: check_hallucination → rewrite oder END
        workflow.add_conditional_edges(
            "check_hallucination",
            self.should_rewrite,
            {"rewrite": "rewrite", "end": END},
        )

        # Rewrite geht zurück zu retrieve
        workflow.add_edge("rewrite", "retrieve")

        return workflow.compile()

    def run(self, query: str) -> AgentState:
        """
        Führt den kompletten Workflow aus.

        Args:
            query: Die Benutzeranfrage.

        Returns:
            Den finalen AgentState.
        """
        graph = self.build_graph()
        initial_state: AgentState = {
            "query": query,
            "retrieved_docs": [],
            "graded_docs": [],
            "answer": "",
            "needs_rewrite": False,
            "hallucination_score": 0.0,
            "iteration": 0,
            "tool_calls": [],
        }
        result = graph.invoke(initial_state)
        return result


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not HAS_LANGGRAPH:
        print("=" * 60)
        print("  LangGraph ist nicht installiert!")
        print("  Installiere mit: uv pip install langgraph")
        print("=" * 60)
        print("\nFühre stattdessen die Pure-Python-Demo aus:")
        print("  python rag_agent.py")
        import sys
        sys.exit(1)

    print("=" * 60)
    print("  LangGraph RAG Agent — Demo")
    print("=" * 60)

    # ── Dokumente ────────────────────────────────────────────
    documents = [
        "Vergütungsvereinbarung 2025: Der Basisfallwert beträgt 4.200 Euro.",
        "Vergütungsvereinbarung 2026: Der Basisfallwert steigt auf 4.350 Euro.",
        "SGB V §87: Die Vergütung der Krankenhäuser richtet sich nach Fallpauschalen.",
        "Krankenhausfinanzierung: DRG-System wurde 2003 eingeführt.",
        "Qualitätsberichte: Krankenhäuser müssen jährlich Qualitätsberichte veröffentlichen.",
        "Pflegepersonaluntergrenzen: Seit 2019 gelten verbindliche Untergrenzen.",
        "Hybrid-DRG: Neue Vergütungsform für bestimmte Leistungen ab 2024.",
        "Notfallversorgung: Reform der Notfallversorgung ist in Planung.",
    ]

    # ── Agent ────────────────────────────────────────────────
    print("\n🤖 Initialisiere LangGraph RAG Agent...")
    agent = RAGAgent(documents, max_iterations=3)

    # ── Queries ──────────────────────────────────────────────
    queries = [
        "Wie hoch ist der Basisfallwert 2025?",
        "Was ist das DRG-System?",
        "Welche Reformen gibt es?",
    ]

    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"🔍 Query: \"{query}\"")
        result = agent.run(query)

        print(f"   📊 Iterationen: {result['iteration']}")
        print(f"   📄 Gefundene Docs: {len(result['retrieved_docs'])}")
        print(f"   ⭐ Bewertete Docs: {len(result['graded_docs'])}")
        print(f"   🎯 Halluzination-Score: {result['hallucination_score']}")
        print(f"   🔄 Rewrite nötig: {result['needs_rewrite']}")
        print(f"\n📝 Antwort:\n{result['answer']}")

    print(f"\n{'─' * 60}")
    print("✅ LangGraph RAG-Demo abgeschlossen!")
    print("   Workflow: retrieve → grade_docs → generate → check_hallucination")
    print("   Conditional Edges: should_generate, should_rewrite")

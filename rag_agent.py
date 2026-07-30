"""
RAG Agent mit LangGraph
=======================
Retrieval-Augmented Generation — Stateful Agent mit Tool-Use.

Architektur:
1. User Query → Retrieve (Qdrant/FAISS)
2. Retrieved Docs + Query → Generate (LLM)
3. Optional: Hallucination Check → Rewrite → Retry

Verwendet LangGraph für den State-Flow.
"""

import operator
from typing import Annotated, TypedDict

import numpy as np

from vector_store import SimpleEmbedder, SimpleVectorStore

# ═══════════════════════════════════════════════════════════════
# State-Definition
# ═══════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    query: str
    retrieved_docs: Annotated[list[str], operator.add]
    context: str
    answer: str
    needs_rewrite: bool
    iteration: int


# ═══════════════════════════════════════════════════════════════
# RAG Pipeline (ohne LangGraph — pure Python für Demo)
# ═══════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Komplette RAG-Pipeline: Retrieve → Generate.

    In Produktion mit LangGraph:
        from langgraph.graph import StateGraph
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("generate", generate_node)
        workflow.add_conditional_edges(...)
    """

    def __init__(self, documents: list[str]):
        self.store = SimpleVectorStore()
        self.embedder = SimpleEmbedder()

        # Dokumente embedden und speichern
        self.embedder.fit(documents)
        embeddings = [self.embedder.embed(doc) for doc in documents]
        self.store.add(documents, embeddings)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieval-Schritt: Query → Embedding → Search."""
        query_emb = self.embedder.embed(query)
        return self.store.search(query_emb, k=k)

    def generate(self, query: str, docs: list[str]) -> str:
        """
        Generation-Schritt (vereinfacht — ohne LLM).
        In Produktion: OpenAI, Claude, lokales LLM.
        """
        if not docs:
            return "Keine relevanten Dokumente gefunden."

        # Template-basierte Antwort (Demo)
        context = "\n".join(f"• {doc}" for doc in docs)
        return (
            f"Basierend auf {len(docs)} Dokumenten:\n\n"
            f"{context}\n\n"
            f"→ Antwort auf: \"{query}\""
        )

    def query(self, query: str, k: int = 3) -> dict:
        """Führt eine komplette RAG-Query aus."""
        docs = self.retrieve(query, k=k)
        answer = self.generate(query, docs)
        return {
            "query": query,
            "retrieved_docs": docs,
            "answer": answer,
        }


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Agent — Demo")
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

    # ── Pipeline ─────────────────────────────────────────────
    print("\n📚 Lade 8 Dokumente...")
    rag = RAGPipeline(documents)

    # ── Queries ──────────────────────────────────────────────
    queries = [
        "Wie hoch ist der Basisfallwert 2025?",
        "Was ist das DRG-System?",
        "Welche Reformen gibt es?",
    ]

    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        result = rag.query(query, k=3)
        print(f"   Gefundene Docs: {len(result['retrieved_docs'])}")
        for doc in result["retrieved_docs"]:
            print(f"   • {doc}")
        print(f"\n📝 Antwort:\n{result['answer']}")

    print("\n✅ RAG-Demo abgeschlossen!")

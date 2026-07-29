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

from typing import TypedDict, List, Annotated
import operator
import numpy as np


# ═══════════════════════════════════════════════════════════════
# State-Definition
# ═══════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    query: str
    retrieved_docs: Annotated[List[str], operator.add]
    context: str
    answer: str
    needs_rewrite: bool
    iteration: int


# ═══════════════════════════════════════════════════════════════
# Einfacher Vector Store (FAISS-ähnlich, in-memory)
# ═══════════════════════════════════════════════════════════════

class SimpleVectorStore:
    """
    Minimaler Vector Store für Demo-Zwecke.
    In Produktion: Qdrant, Pinecone, Weaviate.
    """

    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []

    def add(self, documents: List[str], embeddings: List[np.ndarray]) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[str]:
        """Cosine-Similarity-Suche."""
        if not self.embeddings:
            return []
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scores = []
        for emb in self.embeddings:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            scores.append(float(np.dot(query_norm, emb_norm)))
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k]


# ═══════════════════════════════════════════════════════════════
# Einfacher Embedder (TF-IDF-basiert, kein externes Modell nötig)
# ═══════════════════════════════════════════════════════════════

class SimpleEmbedder:
    """
    TF-IDF-basierter Embedder für Demo-Zwecke.
    In Produktion: OpenAI Embeddings, sentence-transformers, etc.
    """

    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def fit(self, documents: List[str]) -> None:
        """Baut Vokabular und IDF-Werte auf."""
        # Tokenisierung
        tokenized = [doc.lower().split() for doc in documents]
        # Vokabular
        all_tokens = set()
        for tokens in tokenized:
            all_tokens.update(tokens)
        self.vocab = {token: i for i, token in enumerate(sorted(all_tokens))}
        # IDF
        N = len(documents)
        for token in self.vocab:
            df = sum(1 for tokens in tokenized if token in tokens)
            self.idf[token] = np.log((N + 1) / (df + 1)) + 1

    def embed(self, text: str) -> np.ndarray:
        """Erzeugt TF-IDF-Embedding."""
        tokens = text.lower().split()
        vec = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                tf = tokens.count(token) / len(tokens)
                vec[self.vocab[token]] = tf * self.idf.get(token, 1.0)
        return vec


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

    def __init__(self, documents: List[str]):
        self.store = SimpleVectorStore()
        self.embedder = SimpleEmbedder()

        # Dokumente embedden und speichern
        self.embedder.fit(documents)
        embeddings = [self.embedder.embed(doc) for doc in documents]
        self.store.add(documents, embeddings)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieval-Schritt: Query → Embedding → Search."""
        query_emb = self.embedder.embed(query)
        return self.store.search(query_emb, k=k)

    def generate(self, query: str, docs: List[str]) -> str:
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

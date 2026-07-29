"""
Agentic RAG — Multi-Step Retrieval mit Reasoning
================================================
Implementiert das Konzept aus "From Naïve RAG to Agentic RAG" (arXiv:2607.22319).

Erweiterung des bestehenden RAG-Agenten um:
1. Adaptive Retrieval-Strategie (dense + sparse + KG)
2. Multi-Step Reasoning mit Verify-Retrieve-Refine-Zyklus
3. Evidence-Grounded Answer Generation mit Zitaten
"""

from typing import TypedDict, List, Annotated, Optional, Dict, Any
import operator
import numpy as np
import re


class AgenticRAGState(TypedDict):
    """Erweiterter State für agentisches RAG."""
    query: str
    sub_queries: Annotated[List[str], operator.add]
    retrieved_docs: Annotated[List[Dict[str, Any]], operator.add]
    evidence_chains: Annotated[List[Dict[str, Any]], operator.add]
    context: str
    answer: str
    citations: List[str]
    confidence: float
    needs_refinement: bool
    iteration: int
    max_iterations: int


class RetrievalStrategy:
    """
    Adaptive Retrieval-Strategie.
    
    Wählt zwischen:
    - Dense Retrieval (Embedding-Ähnlichkeit)
    - Sparse Retrieval (Keyword/BM25)
    - Graph Retrieval (Knowledge Graph Traversal)
    """

    def __init__(self, strategy: str = "auto"):
        self.strategy = strategy

    def select(self, query: str, previous_results: List[Dict]) -> str:
        """
        Wählt die beste Retrieval-Strategie basierend auf Query-Typ
        und bisherigen Ergebnissen.
        """
        # Heuristik: Faktenfragen → sparse, konzeptionelle → dense
        fact_keywords = ["wann", "wo", "wer", "wie viele", "definition"]
        if any(kw in query.lower() for kw in fact_keywords):
            return "sparse"
        
        # Wenn vorherige Ergebnisse zu wenige → breiter suchen
        if len(previous_results) < 3:
            return "dense"
        
        return "dense"


class EvidenceVerifier:
    """
    Überprüft, ob generierte Antworten durch abgerufene Dokumente
    gestützt werden (Evidence-Grounded Verification).
    """

    def verify(self, answer: str, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """
        Prüft jeden Fakt in der Antwort gegen die Quellen.
        Returns: {verified: bool, unsupported_claims: List[str], citations: List[str]}
        """
        # Extrahiere Behauptungen aus der Antwort
        claims = self._extract_claims(answer)
        
        unsupported = []
        citations = []
        
        for claim in claims:
            found = False
            for doc in retrieved_docs:
                content = doc.get("content", "")
                if self._claim_in_content(claim, content):
                    citations.append(doc.get("source", "unknown"))
                    found = True
                    break
            if not found:
                unsupported.append(claim)
        
        return {
            "verified": len(unsupported) == 0,
            "unsupported_claims": unsupported,
            "citations": list(set(citations)),
        }

    def _extract_claims(self, text: str) -> List[str]:
        """Extrahiere faktische Behauptungen aus Text (Satzebene)."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _claim_in_content(self, claim: str, content: str) -> bool:
        """Prüft, ob eine Behauptung im Quelltext vorkommt (einfache Überlappung)."""
        claim_words = set(claim.lower().split())
        content_words = set(content.lower().split())
        overlap = claim_words & content_words
        return len(overlap) / max(len(claim_words), 1) > 0.3


class QueryDecomposer:
    """
    Zerlegt komplexe Queries in Teilfragen für Multi-Step Retrieval.
    """

    def decompose(self, query: str) -> List[str]:
        """
        Zerlegt eine komplexe Query in atomare Teilfragen.
        In Produktion: LLM-basierte Dekomposition.
        Hier: regelbasierte Heuristik.
        """
        sub_queries = [query]  # Original-Query immer dabei
        
        # Erkenne Vergleichsfragen
        if " vs " in query.lower() or " versus " in query.lower():
            parts = re.split(r'\s+(?:vs|versus)\s+', query, flags=re.IGNORECASE)
            sub_queries.extend([p.strip() for p in parts])
        
        # Erkenne UND-Verknüpfungen
        if " und " in query.lower():
            parts = query.split(" und ")
            sub_queries.extend([p.strip() for p in parts])
        
        return list(set(sub_queries))  # Deduplizieren


class AgenticRAG:
    """
    Agentic RAG: Multi-Step Retrieval mit Reasoning und Verifikation.
    
    Workflow:
    1. Query Decomposition → Sub-Queries
    2. Adaptive Retrieval pro Sub-Query
    3. Evidence Verification
    4. Refinement bei unzureichender Evidenz
    5. Finale Answer Generation mit Zitaten
    """

    def __init__(
        self,
        vector_store,
        embedder,
        llm,
        max_iterations: int = 3,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        self.max_iterations = max_iterations
        self.retrieval_strategy = RetrievalStrategy()
        self.verifier = EvidenceVerifier()
        self.decomposer = QueryDecomposer()

    def run(self, query: str) -> Dict[str, Any]:
        """Führt den vollständigen Agentic RAG-Workflow aus."""
        state: AgenticRAGState = {
            "query": query,
            "sub_queries": [],
            "retrieved_docs": [],
            "evidence_chains": [],
            "context": "",
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "needs_refinement": True,
            "iteration": 0,
            "max_iterations": self.max_iterations,
        }

        # Step 1: Query Decomposition
        state["sub_queries"] = self.decomposer.decompose(query)

        # Step 2-4: Multi-Step Retrieve-Verify-Refine
        while state["needs_refinement"] and state["iteration"] < self.max_iterations:
            state["iteration"] += 1
            
            # Retrieve für jede Sub-Query
            for sub_q in state["sub_queries"]:
                strategy = self.retrieval_strategy.select(sub_q, state["retrieved_docs"])
                docs = self._retrieve(sub_q, strategy)
                state["retrieved_docs"].extend(docs)
            
            # Context aufbauen
            state["context"] = self._build_context(state["retrieved_docs"])
            
            # Generate Answer
            state["answer"] = self._generate(state["query"], state["context"])
            
            # Verify
            verification = self.verifier.verify(state["answer"], state["retrieved_docs"])
            state["citations"] = verification["citations"]
            state["confidence"] = 1.0 - (len(verification["unsupported_claims"]) / max(len(self.verifier._extract_claims(state["answer"])), 1))
            
            # Entscheide: Refinement nötig?
            if verification["verified"] or state["iteration"] >= self.max_iterations:
                state["needs_refinement"] = False
            else:
                # Refinement: Füge unsupported claims als neue Sub-Queries hinzu
                state["sub_queries"] = verification["unsupported_claims"]

        return {
            "answer": state["answer"],
            "citations": state["citations"],
            "confidence": state["confidence"],
            "iterations": state["iteration"],
            "docs_retrieved": len(state["retrieved_docs"]),
        }

    def _retrieve(self, query: str, strategy: str) -> List[Dict[str, Any]]:
        """Führt Retrieval mit der gewählten Strategie aus."""
        query_embedding = self.embedder.embed(query)
        docs = self.vector_store.search(query_embedding, k=5)
        return [{"content": doc, "source": f"doc_{i}", "strategy": strategy} for i, doc in enumerate(docs)]

    def _build_context(self, docs: List[Dict]) -> str:
        """Baut den Kontext aus abgerufenen Dokumenten."""
        return "\n\n".join([d["content"] for d in docs])

    def _generate(self, query: str, context: str) -> str:
        """Generiert Antwort mit Zitaten (in Produktion: LLM-Call)."""
        # Platzhalter — in Produktion wird hier der LLM aufgerufen
        return f"[Agentic RAG] Antwort auf: {query}\n\nKontext-basierte Generierung mit {len(context)} Zeichen Kontext."

"""
Gemeinsame Vector-Store- und Embedder-Klassen für RAG-Demos.

Wiederverwendet von rag_agent.py und langgraph_agent.py.
"""

import numpy as np


class SimpleVectorStore:
    """Minimaler Vector Store mit Cosine-Similarity."""

    def __init__(self):
        self.documents: list[str] = []
        self.embeddings: list[np.ndarray] = []

    def add(self, documents: list[str], embeddings: list[np.ndarray]) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[str]:
        if not self.embeddings:
            return []
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scores = []
        for emb in self.embeddings:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            scores.append(float(np.dot(query_norm, emb_norm)))
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k]


class SimpleEmbedder:
    """TF-IDF-basierter Embedder."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[str]) -> None:
        tokenized = [doc.lower().split() for doc in documents]
        all_tokens: set[str] = set()
        for tokens in tokenized:
            all_tokens.update(tokens)
        self.vocab = {token: i for i, token in enumerate(sorted(all_tokens))}
        n_docs = len(documents)
        for token in self.vocab:
            df = sum(1 for tokens in tokenized if token in tokens)
            self.idf[token] = np.log((n_docs + 1) / (df + 1)) + 1

    def embed(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        vec = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                tf = tokens.count(token) / len(tokens)
                vec[self.vocab[token]] = tf * self.idf.get(token, 1.0)
        return vec

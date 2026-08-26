"""
Gemeinsame Vector-Store- und Embedder-Klassen für RAG-Demos.

Wiederverwendet von rag_agent.py und langgraph_agent.py.
"""

import hashlib
import os
import re

import numpy as np

DEFAULT_PERSIST_DIR = os.environ.get("VECTOR_DB_PATH", "./data/chroma")
DEFAULT_COLLECTION = "documents"


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


class HashEmbedder:
    """
    Hashing-basierter Embedder mit fester Dimension.

    Anders als SimpleEmbedder (TF-IDF, Vokabular wächst mit dem Korpus)
    hat HashEmbedder eine feste Vektor-Dimension unabhängig vom Korpus —
    Voraussetzung für einen Vector Store, der über mehrere Sessions und
    Prozess-Neustarts hinweg gültig bleibt (persistente Speicherung).
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        words = re.findall(r"\b\w+\b", text.lower())
        vec = np.zeros(self.dim)
        if not words:
            return vec
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0 / (1.0 + i * 0.01)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class PersistentVectorStore:
    """
    Vector Store mit dauerhafter Speicherung auf Disk (ChromaDB).

    Dokumente und Embeddings überleben Prozess-Neustarts und
    Container-Redeploys, solange `persist_directory` auf ein
    persistentes Volume zeigt (siehe Dockerfile/docker-compose).
    """

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        import chromadb

        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        documents: list[str],
        embeddings: list[np.ndarray],
        metadatas: list[dict] | None = None,
    ) -> None:
        if not documents:
            return
        start_id = self.collection.count()
        ids = [f"doc-{start_id + i}" for i in range(len(documents))]
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=[np.asarray(e).tolist() for e in embeddings],
            metadatas=metadatas,
        )

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[str]:
        if self.collection.count() == 0:
            return []
        k = min(k, self.collection.count())
        result = self.collection.query(
            query_embeddings=[np.asarray(query_embedding).tolist()],
            n_results=k,
        )
        return result["documents"][0] if result["documents"] else []

    def search_with_scores(
        self, query_embedding: np.ndarray, k: int = 3
    ) -> list[tuple[str, float, dict]]:
        """Wie search(), gibt zusätzlich Cosine-Similarity und Metadaten zurück."""
        if self.collection.count() == 0:
            return []
        k = min(k, self.collection.count())
        result = self.collection.query(
            query_embeddings=[np.asarray(query_embedding).tolist()],
            n_results=k,
        )
        docs = result["documents"][0] if result["documents"] else []
        distances = result["distances"][0] if result.get("distances") else [0.0] * len(docs)
        metadatas = result["metadatas"][0] if result.get("metadatas") else [{}] * len(docs)
        # hnsw:space=cosine → distance = 1 - cosine_similarity
        return [
            (doc, 1.0 - dist, meta or {})
            for doc, dist, meta in zip(docs, distances, metadatas)
        ]

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)

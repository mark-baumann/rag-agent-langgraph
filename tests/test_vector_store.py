"""
Tests für die gemeinsamen Vector-Store-Klassen (vector_store.py).

Testet:
- HashEmbedder: feste Dimension, Determinismus
- PersistentVectorStore: add, search, Persistenz über Prozess-Neustarts hinweg
"""

import numpy as np
import pytest

from vector_store import HashEmbedder, PersistentVectorStore, SimpleEmbedder, SimpleVectorStore


class TestSimpleVectorStore:
    def test_add_and_search(self):
        store = SimpleVectorStore()
        store.add(["a", "b"], [np.array([1.0, 0.0]), np.array([0.0, 1.0])])
        results = store.search(np.array([1.0, 0.0]), k=1)
        assert results == ["a"]

    def test_search_empty_store(self):
        store = SimpleVectorStore()
        assert store.search(np.array([1.0, 0.0])) == []


class TestSimpleEmbedder:
    def test_fit_builds_vocab(self):
        embedder = SimpleEmbedder()
        embedder.fit(["hallo welt", "welt der programmierung"])
        assert "hallo" in embedder.vocab
        assert "welt" in embedder.vocab


class TestHashEmbedder:
    def test_fixed_dimension(self):
        embedder = HashEmbedder(dim=64)
        vec = embedder.embed("ein beliebiger text")
        assert vec.shape == (64,)

    def test_deterministic(self):
        embedder = HashEmbedder(dim=64)
        assert np.array_equal(embedder.embed("gleicher text"), embedder.embed("gleicher text"))

    def test_empty_text(self):
        embedder = HashEmbedder(dim=32)
        vec = embedder.embed("")
        assert np.count_nonzero(vec) == 0

    def test_dimension_independent_of_corpus(self):
        """Im Gegensatz zu SimpleEmbedder bleibt die Dimension über mehrere
        unabhängige Aufrufe/Korpora hinweg konstant — Voraussetzung für
        eine Vector-DB, die über Sessions hinweg persistiert."""
        embedder = HashEmbedder(dim=64)
        assert embedder.embed("kurzer text").shape == embedder.embed(
            "ein deutlich längerer text mit vielen weiteren wörtern"
        ).shape


class TestPersistentVectorStore:
    def test_add_and_search(self, tmp_path):
        store = PersistentVectorStore(persist_directory=str(tmp_path / "chroma"))
        embedder = HashEmbedder(dim=32)
        docs = ["Katzen sind Haustiere.", "Python ist eine Programmiersprache."]
        store.add(docs, [embedder.embed(d) for d in docs])

        assert store.count() == 2
        results = store.search(embedder.embed("Katzen"), k=1)
        assert results == ["Katzen sind Haustiere."]

    def test_search_with_scores_returns_metadata(self, tmp_path):
        store = PersistentVectorStore(persist_directory=str(tmp_path / "chroma"))
        embedder = HashEmbedder(dim=32)
        store.add(["Dokument A"], [embedder.embed("Dokument A")], metadatas=[{"source": "a.pdf"}])

        results = store.search_with_scores(embedder.embed("Dokument A"), k=1)
        assert len(results) == 1
        doc, score, meta = results[0]
        assert doc == "Dokument A"
        assert meta["source"] == "a.pdf"
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_persists_across_process_restarts(self, tmp_path):
        """Simuliert einen Container-Neustart: neue Client-Instanz, gleicher Pfad."""
        persist_dir = str(tmp_path / "chroma")
        embedder = HashEmbedder(dim=32)

        store1 = PersistentVectorStore(persist_directory=persist_dir)
        store1.add(["Dauerhaft gespeicherter Text."], [embedder.embed("Dauerhaft gespeicherter Text.")])
        del store1

        store2 = PersistentVectorStore(persist_directory=persist_dir)
        assert store2.count() == 1
        assert store2.search(embedder.embed("Dauerhaft"), k=1) == ["Dauerhaft gespeicherter Text."]

    def test_clear(self, tmp_path):
        store = PersistentVectorStore(persist_directory=str(tmp_path / "chroma"))
        embedder = HashEmbedder(dim=32)
        store.add(["a", "b"], [embedder.embed("a"), embedder.embed("b")])
        assert store.count() == 2

        store.clear()
        assert store.count() == 0

    def test_search_empty_store(self, tmp_path):
        store = PersistentVectorStore(persist_directory=str(tmp_path / "chroma"))
        assert store.search(np.zeros(32)) == []

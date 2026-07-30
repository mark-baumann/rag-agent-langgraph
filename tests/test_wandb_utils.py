"""
Tests für wandb_utils.py — W&B Experiment Tracking für RAG Agent.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wandb_utils import WANDB_AVAILABLE, WandBTracker


class TestWandBTracker:
    """Tests für den WandBTracker."""

    def test_initialization_offline(self):
        """Tracker sollte im Offline-Modus initialisieren."""
        tracker = WandBTracker(
            project="test-rag",
            config={"k": 3, "embedder": "tf-idf"},
            tags=["test", "rag"],
            group="test-group",
            job_type="test",
            notes="Test-Run",
            offline=True,
        )
        if WANDB_AVAILABLE:
            assert tracker.is_active
            assert tracker.run is not None
        else:
            assert not tracker.is_active
        tracker.finish()

    def test_log_retrieval(self):
        """Retrieval-Metriken sollten ohne Fehler geloggt werden."""
        tracker = WandBTracker(project="test-rag", offline=True)
        if tracker.is_active:
            tracker.log_retrieval(
                query="Wie hoch ist der Basisfallwert?",
                retrieved_docs=["Dokument 1", "Dokument 2"],
                k=3,
                retrieval_time_ms=12.5,
            )
        tracker.finish()

    def test_log_embedding_stats(self):
        """Embedding-Statistiken sollten geloggt werden."""
        tracker = WandBTracker(project="test-rag", offline=True)
        if tracker.is_active:
            tracker.log_embedding_stats(vocab_size=100, embedding_dim=50)
        tracker.finish()

    def test_log_query_result(self):
        """Query-Ergebnisse sollten geloggt werden."""
        tracker = WandBTracker(project="test-rag", offline=True)
        if tracker.is_active:
            tracker.log_query_result(
                query="Test-Query",
                answer="Test-Antwort mit Inhalt",
                num_docs=3,
            )
        tracker.finish()

    def test_finish_cleans_up(self):
        """finish() sollte den Run beenden und safe für doppelte Aufrufe sein."""
        tracker = WandBTracker(project="test-rag", offline=True)
        tracker.finish()
        tracker.finish()

    def test_multiple_queries(self):
        """Mehrere Queries sollten korrekt getrackt werden."""
        tracker = WandBTracker(project="test-rag", offline=True)
        if tracker.is_active:
            for i in range(3):
                tracker.log_retrieval(
                    query=f"Query {i}",
                    retrieved_docs=[f"Doc {j}" for j in range(i + 1)],
                    k=3,
                    retrieval_time_ms=10.0 + i,
                )
        tracker.finish()

"""
W&B Experiment Tracking für RAG Agent
=====================================
Integriert Weights & Biases in die RAG-Pipeline.
Loggt Retrieval-Metriken, Query-Ergebnisse und Embedding-Qualität.

Verwendung:
    from wandb_utils import WandBTracker
    tracker = WandBTracker(project="rag-agent", config={...})
    tracker.log_retrieval(query, docs, k=3)
    tracker.finish()
"""

import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Gekapselter W&B-Tracker für RAG-Pipelines.

    Features:
    - Retrieval-Qualität tracken (Anzahl Docs, Relevanz)
    - Query-Latenz messen
    - Embedding-Dimensionen loggen
    - Automatischer Offline-Modus
    """

    def __init__(
        self,
        project: str = "rag-agent",
        config: dict | None = None,
        tags: list | None = None,
        group: str | None = None,
        job_type: str = "eval",
        notes: str | None = None,
        offline: bool = False,
    ):
        self.project = project
        self.run = None
        self._query_count = 0
        self._total_retrieval_time = 0.0

        if WANDB_AVAILABLE:
            try:
                mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
                self.run = wandb.init(
                    project=project,
                    config=config or {},
                    mode=mode,
                    tags=tags or ["rag", "retrieval", "nlp"],
                    group=group,
                    job_type=job_type,
                    notes=notes,
                    dir="wandb_runs",
                )
                if mode == "online":
                    try:
                        import subprocess
                        git_commit = subprocess.check_output(
                            ["git", "rev-parse", "--short", "HEAD"],
                            stderr=subprocess.DEVNULL,
                        ).decode().strip()
                        self.log({"git_commit": git_commit})
                    except Exception:
                        pass
                print(f"📊 W&B initialisiert (mode={mode}, project={project})")
            except Exception as e:
                print(f"⚠️  W&B-Init fehlgeschlagen: {e}")

    def log(self, metrics: dict, step: int | None = None):
        """Loggt Metriken zu W&B."""
        if self.run:
            self.run.log(metrics, step=step)

    def log_retrieval(
        self,
        query: str,
        retrieved_docs: list[str],
        k: int = 3,
        retrieval_time_ms: float = 0.0,
    ):
        """Loggt Retrieval-Ergebnisse."""
        self._query_count += 1
        self._total_retrieval_time += retrieval_time_ms

        metrics = {
            "retrieval/query_count": self._query_count,
            "retrieval/docs_found": len(retrieved_docs),
            "retrieval/k_requested": k,
            "retrieval/recall_at_k": len(retrieved_docs) / max(k, 1),
            "retrieval/time_ms": retrieval_time_ms,
            "retrieval/avg_time_ms": self._total_retrieval_time / self._query_count,
        }
        self.log(metrics)

        # Logge Query + Docs als Tabelle
        if self.run and retrieved_docs:
            table = wandb.Table(columns=["query", "rank", "document"])
            for i, doc in enumerate(retrieved_docs):
                table.add_data(query, i + 1, doc[:200])
            self.run.log({"retrieval/results": table})

    def log_embedding_stats(self, vocab_size: int, embedding_dim: int):
        """Loggt Embedding-Statistiken."""
        self.log({
            "embedding/vocab_size": vocab_size,
            "embedding/dim": embedding_dim,
        })

    def log_query_result(self, query: str, answer: str, num_docs: int):
        """Loggt ein komplettes Query-Ergebnis."""
        self._query_count += 1
        self.log({
            "query/num_docs_used": num_docs,
            "query/answer_length": len(answer),
        })

    def finish(self):
        """Beendet den W&B-Run."""
        if self.run:
            self.run.finish()

    @property
    def is_active(self) -> bool:
        return self.run is not None

# RAG Agent mit LangGraph

**Retrieval-Augmented Generation** — Stateful Agent mit Tool-Use.

## Architektur

```
User Query → Retrieve (Vector Store) → Generate (LLM) → Answer
                  ↑                        ↓
                  └── Optional: Rewrite ← Hallucination Check
```

## Komponenten

| Komponente | Beschreibung |
|---|---|
| `SimpleVectorStore` | In-Memory Vector Store mit Cosine-Similarity-Suche |
| `SimpleEmbedder` | TF-IDF-basierter Embedder (kein externes Modell nötig) |
| `RAGPipeline` | Komplette Pipeline: Retrieve → Generate |
| `RAGState` | TypedDict für LangGraph State-Flow |

## Installation

```bash
git clone https://github.com/mark-baumann/rag-agent-langgraph.git
cd rag-agent-langgraph
python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Verwendung

```python
from rag_agent import RAGPipeline

documents = [
    "Dokument 1: ...",
    "Dokument 2: ...",
]

rag = RAGPipeline(documents)
result = rag.query("Deine Frage", k=3)
print(result["answer"])
```

Oder direkt:

```bash
python rag_agent.py
```

## Tests

```bash
pytest tests/ -v
```

## Produktionsreif

In Produktion ersetzen durch:

- **Vector Store:** Qdrant, Pinecone, Weaviate
- **Embedder:** OpenAI Embeddings, sentence-transformers
- **LLM:** OpenAI, Claude, lokales Modell
- **Orchestrierung:** LangGraph `StateGraph` mit Conditional Edges

## Lizenz

MIT

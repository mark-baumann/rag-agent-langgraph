# 🔍 RAG Agent mit LangGraph

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Agent-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

**Retrieval-Augmented Generation mit LangGraph** — Stateful RAG-Pipeline mit Tool-Use, Halluzinationserkennung und agentischem Multi-Step-Reasoning.

## 📋 Beschreibung

Dieses Repository implementiert eine vollständige RAG-Pipeline auf Basis von LangGraph — von der einfachen Retrieve-Generate-Schleife bis zum agentischen Multi-Step-Retrieval mit adaptiver Strategieauswahl. Drei aufeinander aufbauende Implementierungen zeigen die Evolution von Naïve RAG zu Agentic RAG.

- **RAG Agent** — Grundlegende Retrieve → Generate Pipeline mit Vector Store
- **LangGraph Agent** — Stateful Workflow mit Retrieve → Grade → Generate → Hallucination Check
- **Agentic RAG** — Multi-Step Reasoning mit Verify-Retrieve-Refine-Zyklus und Evidence-Grounding

## ✨ Features

- 🔄 **Stateful Workflow** — LangGraph StateGraph mit typisiertem State und Conditional Edges
- 📚 **Vector Store** — In-Memory Vector Store mit Cosine-Similarity-Suche
- 🛠️ **Tool-Use** — Calculator, Web Search Stub für erweiterte Agent-Fähigkeiten
- ✅ **Halluzinationserkennung** — Automatischer Check und Rewrite-Loop bei unsicheren Antworten
- 🎯 **Adaptive Retrieval** — Dense, Sparse und Graph-basierte Retrieval-Strategien
- 📊 **W&B Tracking** — Experiment-Tracking für RAG-Pipeline-Performance
- 🖥️ **Streamlit-App** — Interaktive Demo aller drei RAG-Varianten
- 🧪 **Test-Suite** — pytest-Tests für alle Agent-Varianten

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/mark-baumann/rag-agent-langgraph.git
cd rag-agent-langgraph

# Virtuelle Umgebung erstellen
python3 -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Optional: LangGraph (für vollen Workflow)
pip install langgraph
```

## 🎮 Nutzung

### Streamlit-App

```bash
streamlit run app.py
```

Die App bietet drei Modi:
1. **RAG Agent** — Einfache Retrieve-Generate-Pipeline mit konfigurierbarem Vector Store
2. **LangGraph Agent** — Stateful Workflow mit Halluzinationserkennung und Tool-Use
3. **Agentic RAG** — Multi-Step Reasoning mit adaptiver Retrieval-Strategie

### Direkt per Python

```python
from rag_agent import RAGAgent, SimpleVectorStore

# Vector Store mit Dokumenten befüllen
store = SimpleVectorStore()
store.add(["Python ist eine Programmiersprache", "PyTorch ist ein ML-Framework"], 
          embeddings=[[0.1, 0.2], [0.3, 0.4]])

# RAG-Agent ausführen
agent = RAGAgent(store)
result = agent.query("Was ist Python?")
print(result["answer"])
```

### Tests

```bash
pytest tests/ -v
```

## 🏗️ Tech-Stack

| Komponente | Technologie |
|---|---|
| **Sprache** | Python 3.10+ |
| **Orchestrierung** | LangGraph (StateGraph) |
| **Embeddings** | NumPy (In-Memory) |
| **UI** | Streamlit |
| **Tracking** | Weights & Biases |
| **Testing** | pytest |

## 📁 Projektstruktur

```
rag-agent-langgraph/
├── rag_agent.py            # Basis RAG-Agent mit Vector Store
├── langgraph_agent.py      # LangGraph Stateful Workflow
├── agentic_rag.py          # Agentic RAG mit Multi-Step Reasoning
├── app.py                  # Streamlit-App
├── wandb_utils.py          # W&B Integration
└── tests/
    ├── test_rag_agent.py
    ├── test_langgraph_agent.py
    ├── test_agentic_rag.py
    └── test_wandb_utils.py
```

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)

---

*Für Fragen oder Beiträge: Issue erstellen oder Pull Request öffnen.*

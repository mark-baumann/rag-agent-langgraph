"""
Streamlit-App: RAG Agent mit LangGraph
======================================
PDF hochladen → embedden → Fragen stellen, Retrieval visualisieren.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag_agent import RAGPipeline, SimpleVectorStore, SimpleEmbedder

st.set_page_config(
    page_title="RAG Agent — LangGraph",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Agent mit LangGraph")
st.markdown("### PDF hochladen → embedden → Fragen stellen → Retrieval visualisieren")

# ── Session State ──
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ──
st.sidebar.header("📂 Dokumenten-Quelle")

doc_source = st.sidebar.radio(
    "Quelle wählen",
    ["📄 PDF hochladen", "📝 Text eingeben", "📚 Demo-Dokumente"],
)

# ═══════════════════════════════════════════════════════════════
# PDF UPLOAD
# ═══════════════════════════════════════════════════════════════
if doc_source == "📄 PDF hochladen":
    st.header("📄 PDF-Dokumente hochladen")

    uploaded_files = st.file_uploader(
        "PDF-Dateien auswählen",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        documents = []

        for uploaded_file in uploaded_files:
            file_ext = uploaded_file.name.split(".")[-1].lower()

            if file_ext == "pdf":
                try:
                    # PDF mit PyMuPDF (fitz) extrahieren
                    import fitz  # PyMuPDF

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    doc = fitz.open(tmp_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()

                    # In Absätze aufteilen
                    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
                    documents.extend(paragraphs)

                    doc.close()
                    os.unlink(tmp_path)

                    st.success(f"✅ **{uploaded_file.name}**: {len(paragraphs)} Absätze extrahiert")

                except ImportError:
                    st.error(
                        "⚠️ PyMuPDF (fitz) nicht installiert. "
                        "Installiere mit: `uv pip install pymupdf`"
                    )
                    # Fallback: Roh-Text
                    text = uploaded_file.read().decode("utf-8", errors="replace")
                    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
                    documents.extend(paragraphs)

            elif file_ext == "txt":
                text = uploaded_file.read().decode("utf-8", errors="replace")
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
                documents.extend(paragraphs)
                st.success(f"✅ **{uploaded_file.name}**: {len(paragraphs)} Absätze geladen")

        if documents:
            st.session_state.documents = documents

            if st.button("🔨 Embeddings erstellen & Pipeline starten", type="primary",
                         use_container_width=True):
                with st.spinner("Erstelle Embeddings und baue RAG-Pipeline..."):
                    st.session_state.rag_pipeline = RAGPipeline(documents)
                st.success(f"✅ Pipeline bereit! **{len(documents)}** Dokumente embeddet.")
                st.rerun()

# ═══════════════════════════════════════════════════════════════
# TEXT EINGEBEN
# ═══════════════════════════════════════════════════════════════
elif doc_source == "📝 Text eingeben":
    st.header("📝 Eigene Texte eingeben")

    text_input = st.text_area(
        "Füge deine Texte ein (ein Absatz pro Zeile, Leerzeile trennt Dokumente)",
        value="",
        height=300,
        placeholder="Dokument 1: Dies ist der erste Text...\n\nDokument 2: Ein weiterer Text...",
    )

    if text_input.strip():
        documents = [p.strip() for p in text_input.split("\n\n") if len(p.strip()) > 10]

        if st.button("🔨 Embeddings erstellen & Pipeline starten", type="primary",
                     use_container_width=True):
            with st.spinner("Erstelle Embeddings..."):
                st.session_state.documents = documents
                st.session_state.rag_pipeline = RAGPipeline(documents)
            st.success(f"✅ Pipeline bereit! **{len(documents)}** Dokumente embeddet.")
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# DEMO-DOKUMENTE
# ═══════════════════════════════════════════════════════════════
elif doc_source == "📚 Demo-Dokumente":
    st.header("📚 Demo-Dokumente (Gesundheitswesen)")

    demo_docs = [
        "Vergütungsvereinbarung 2025: Der Basisfallwert beträgt 4.200 Euro.",
        "Vergütungsvereinbarung 2026: Der Basisfallwert steigt auf 4.350 Euro.",
        "SGB V §87: Die Vergütung der Krankenhäuser richtet sich nach Fallpauschalen.",
        "Krankenhausfinanzierung: Das DRG-System wurde 2003 eingeführt.",
        "Qualitätsberichte: Krankenhäuser müssen jährlich Qualitätsberichte veröffentlichen.",
        "Pflegepersonaluntergrenzen: Seit 2019 gelten verbindliche Untergrenzen.",
        "Hybrid-DRG: Neue Vergütungsform für bestimmte Leistungen ab 2024.",
        "Notfallversorgung: Reform der Notfallversorgung ist in Planung.",
        "Krankenhausstrukturfonds: Förderung von Strukturveränderungen seit 2016.",
        "MDK-Reform: Der Medizinische Dienst wurde 2020 reformiert.",
        "Telematikinfrastruktur: Alle Krankenhäuser müssen bis 2025 angebunden sein.",
        "Klimaschutz: Krankenhäuser müssen Klimaschutzpläne vorlegen.",
    ]

    st.markdown("**Verfügbare Dokumente:**")
    for i, doc in enumerate(demo_docs, 1):
        st.markdown(f"{i}. {doc}")

    if st.button("🔨 Demo-Pipeline starten", type="primary", use_container_width=True):
        with st.spinner("Erstelle Embeddings..."):
            st.session_state.documents = demo_docs
            st.session_state.rag_pipeline = RAGPipeline(demo_docs)
        st.success(f"✅ Demo-Pipeline bereit! **{len(demo_docs)}** Dokumente embeddet.")
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# FRAGEN STELLEN
# ═══════════════════════════════════════════════════════════════
st.markdown("---")

if st.session_state.rag_pipeline is not None:
    st.header("💬 Fragen stellen")

    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_input(
            "Deine Frage",
            placeholder="z.B. Wie hoch ist der Basisfallwert 2025?",
            key="query_input",
        )

        k_docs = st.slider("Anzahl abzurufender Dokumente (k)", 1, 10, 3)

        if st.button("🔍 Suchen", type="primary", use_container_width=True, disabled=not query.strip()):
            with st.spinner("Suche relevante Dokumente..."):
                result = st.session_state.rag_pipeline.query(query, k=k_docs)

            st.session_state.chat_history.append({
                "query": query,
                "result": result,
            })

    with col2:
        st.subheader("📊 Pipeline-Status")
        st.metric("Dokumente im Store", len(st.session_state.documents))
        st.metric("Vokabular-Größe",
                  len(st.session_state.rag_pipeline.embedder.vocab))

    # Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 Antworten")

        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {entry['query'][:80]}...", expanded=(i == 0)):
                result = entry["result"]

                st.markdown(f"**🔍 Query:** _{result['query']}_")

                # Retrieval visualisieren
                st.markdown("**📚 Abgerufene Dokumente:**")
                for j, doc in enumerate(result["retrieved_docs"]):
                    st.markdown(f"*{j + 1}.* {doc}")

                st.markdown("**📝 Antwort:**")
                st.info(result["answer"])

    # Retrieval-Visualisierung
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📈 Retrieval-Visualisierung")

        latest = st.session_state.chat_history[-1]["result"]
        docs = latest["retrieved_docs"]

        if docs:
            # Ähnlichkeits-Scores berechnen
            query_emb = st.session_state.rag_pipeline.embedder.embed(latest["query"])
            scores = []
            for doc in docs:
                doc_emb = st.session_state.rag_pipeline.embedder.embed(doc)
                q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
                d_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-8)
                scores.append(float(np.dot(q_norm, d_norm)))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Balkendiagramm: Relevanz-Scores
            labels = [f"Doc {i + 1}" for i in range(len(docs))]
            colors = plt.cm.YlOrRd(np.array(scores) / max(scores) if max(scores) > 0 else scores)
            ax1.barh(labels, scores, color=colors, edgecolor='white')
            ax1.set_xlabel("Cosine-Ähnlichkeit")
            ax1.set_title("Dokument-Relevanz (Cosine-Similarity)")
            for i, (label, score) in enumerate(zip(labels, scores)):
                ax1.text(score + 0.01, i, f"{score:.3f}", va='center')

            # Heatmap: Query vs Docs
            all_terms = list(st.session_state.rag_pipeline.embedder.vocab.keys())[:20]
            if all_terms:
                query_vec = np.array([
                    st.session_state.rag_pipeline.embedder.embed(latest["query"])[
                        st.session_state.rag_pipeline.embedder.vocab.get(t, 0)
                    ] if t in st.session_state.rag_pipeline.embedder.vocab else 0
                    for t in all_terms
                ])

                doc_vecs = []
                for doc in docs:
                    dv = np.array([
                        st.session_state.rag_pipeline.embedder.embed(doc)[
                            st.session_state.rag_pipeline.embedder.vocab.get(t, 0)
                        ] if t in st.session_state.rag_pipeline.embedder.vocab else 0
                        for t in all_terms
                    ])
                    doc_vecs.append(dv)

                matrix = np.array([query_vec] + doc_vecs)
                im = ax2.imshow(matrix, aspect='auto', cmap='YlOrRd')
                ax2.set_xticks(range(len(all_terms)))
                ax2.set_xticklabels(all_terms, rotation=45, ha='right', fontsize=8)
                ax2.set_yticks(range(len(docs) + 1))
                ax2.set_yticklabels(["Query"] + [f"Doc {i + 1}" for i in range(len(docs))])
                ax2.set_title("TF-IDF Term-Matrix (Top-20 Terme)")
                plt.colorbar(im, ax=ax2)

            st.pyplot(fig)
            plt.close(fig)

else:
    st.info("👆 Lade zuerst Dokumente hoch oder wähle Demo-Dokumente, um die Pipeline zu starten.")

# ═══════════════════════════════════════════════════════════════
# ARCHITEKTUR-DIAGRAMM
# ═══════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
with st.sidebar.expander("🏗️ RAG-Architektur", expanded=False):
    st.markdown("""
    ```
    ┌─────────────┐
    │  PDF Upload │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Text-      │
    │  Extraktion │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Chunking   │
    │  (Absätze)  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Embedding  │
    │  (TF-IDF)   │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Vector     │
    │  Store      │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Query      │
    │  Embedding  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Similarity │
    │  Search     │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Context +  │
    │  Generate   │
    └─────────────┘
    ```
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("📁 **Repo:** [rag-agent-langgraph](https://github.com/mark-baumann/rag-agent-langgraph)")
st.sidebar.markdown("🐍 **Python 3.13** · **Streamlit** · **TF-IDF + Cosine-Sim**")

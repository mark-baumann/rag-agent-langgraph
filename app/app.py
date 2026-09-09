"""
Streamlit-App: RAG Agent LangGraph
==================================
PDF hochladen → embedden → dauerhaft in Vector-DB speichern → Fragen stellen,
Retrieval visualisieren. Gespeicherte Dokumente überleben App-Neustarts und
Redeploys (Vector-DB liegt auf einem persistenten Volume).
"""

import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf_text import chunk_text, extract_text_from_pdf, reindex_poisoned_documents  # noqa: E402
from vector_store import DEFAULT_PERSIST_DIR, HashEmbedder, PersistentVectorStore  # noqa: E402

EMBED_DIM = 256

# Original-PDFs werden neben der Vector-DB auf demselben persistenten
# Volume abgelegt, damit sie im Browser angezeigt oder heruntergeladen
# werden können (die Vector-DB enthält nur Text-Chunks, keine Rohdaten).
DOCS_DIR = Path(DEFAULT_PERSIST_DIR).parent / "documents"
VOLUME_DOCS_DIR = Path(DEFAULT_PERSIST_DIR) / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
VOLUME_DOCS_DIR.mkdir(parents=True, exist_ok=True)
DOC_SEARCH_DIRS = [DOCS_DIR, VOLUME_DOCS_DIR]

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Agent — LangGraph",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Agent — LangGraph")
st.markdown("PDF hochladen · Embedding · dauerhafte Speicherung · Fragen stellen · Retrieval visualisieren")


@st.cache_resource
def bootstrap():
    """Öffnet die Vector-DB und ersetzt alte PyMuPDF-Fehler-Chunks."""
    store = PersistentVectorStore()
    embedder = HashEmbedder(dim=EMBED_DIM)
    stats = reindex_poisoned_documents(store, embedder, DOC_SEARCH_DIRS)
    return store, embedder, stats


store, embedder, reindex_stats = bootstrap()


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📄 PDF-Upload")
    uploaded_file = st.file_uploader("PDF-Datei auswählen", type=["pdf"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        chunk_size = st.slider("Chunk-Größe (Wörter)", 100, 1000, 500, 50)
        overlap = st.slider("Überlappung (Wörter)", 0, 300, 100, 25)

        if st.button("💾 PDF verarbeiten & dauerhaft speichern", type="primary"):
            with st.spinner("📄 Extrahiere Text aus PDF..."):
                text = extract_text_from_pdf(file_bytes)

            if not text.strip():
                st.error(
                    "⚠️ Kein Text aus dieser PDF extrahierbar. "
                    "Weder Text-Layer noch OCR (Tesseract) lieferten Text. "
                    "Das Dokument ist möglicherweise beschädigt oder enthält "
                    "nur nicht-erkennbare Grafiken."
                )
            else:
                with st.spinner("✂️ Erstelle Chunks..."):
                    chunks = chunk_text(text, chunk_size, overlap)

                with st.spinner("🧮 Berechne Embeddings & speichere in Vector-DB..."):
                    embeddings = [embedder.embed(c) for c in chunks]
                    metadatas = [{"source": uploaded_file.name} for _ in chunks]
                    store.add(chunks, embeddings, metadatas=metadatas)

                safe_name = Path(uploaded_file.name).name
                for docs_dir in DOC_SEARCH_DIRS:
                    docs_dir.mkdir(parents=True, exist_ok=True)
                    (docs_dir / safe_name).write_bytes(file_bytes)

                st.success(f"✅ {len(chunks)} Chunks dauerhaft in der Vector-DB gespeichert!")
                st.metric("Neue Chunks", len(chunks))
                st.metric("Chunks insgesamt (Vector-DB)", store.count())
                st.metric("Textlänge", f"{len(text):,} Zeichen")

                # Zeige Text-Vorschau
                with st.expander("📝 Text-Vorschau (erste 1000 Zeichen)"):
                    st.text(text[:1000])

with col2:
    st.subheader("❓ Frage stellen")
    st.caption("Durchsucht alle jemals hochgeladenen PDFs — auch nach einem Neustart der App.")

    if store.count() > 0:
        query = st.text_input("Deine Frage", placeholder="Worum geht es in den Dokumenten?")

        top_k = st.slider("Anzahl Ergebnisse (Top-K)", 1, 10, 5)

        if st.button("🔍 Suchen", type="primary") and query:
            with st.spinner("🔍 Retrieval läuft..."):
                query_emb = embedder.embed(query)
                results = store.search_with_scores(query_emb, k=top_k)

            st.divider()
            st.subheader(f"📊 Top-{len(results)} Ergebnisse")

            # ── Visualisierung: Ähnlichkeits-Balken ──────────
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 3))
            chunk_labels = [f"Chunk {i}" for i in range(len(results))]
            sim_values = [r[1] for r in results]
            colors = plt.cm.Blues(np.array(sim_values) / max(sim_values) if max(sim_values) > 0 else 1)
            bars = ax.barh(chunk_labels[::-1], sim_values[::-1], color=colors[::-1])
            ax.set_xlabel("Kosinus-Ähnlichkeit")
            ax.set_title(f"Retrieval-Ergebnisse für: '{query[:50]}...'")
            for bar, val in zip(bars, sim_values[::-1]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center")
            st.pyplot(fig)

            # ── Ergebnisse im Detail ─────────────────────────
            for i, (text, sim, meta) in enumerate(results):
                source = meta.get("source", "unbekannt")
                with st.expander(f"📌 Chunk {i} · {source} — Ähnlichkeit: {sim:.4f}", expanded=(i == 0)):
                    # Highlight relevante Wörter
                    query_words = set(re.findall(r'\b\w+\b', query.lower()))
                    highlighted = text
                    for word in query_words:
                        if len(word) > 2:
                            highlighted = re.sub(
                                f'\\b({re.escape(word)})\\b',
                                r'**\1**',
                                highlighted,
                                flags=re.IGNORECASE,
                            )
                    st.markdown(highlighted[:1000])

            # ── Retrieval-Statistiken ────────────────────────
            st.divider()
            st.subheader("📈 Retrieval-Statistiken")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Durchschn. Ähnlichkeit", f"{np.mean(sim_values):.4f}")
            with col_b:
                st.metric("Max. Ähnlichkeit", f"{max(sim_values):.4f}")
            with col_c:
                st.metric("Chunks > 0.1", f"{sum(1 for s in sim_values if s > 0.1)}/{len(results)}")

    else:
        st.info("👈 Lade zuerst eine PDF-Datei hoch und klicke auf 'PDF verarbeiten & dauerhaft speichern'.")

# ═══════════════════════════════════════════════════════════════
# Gespeicherte Dokumente: anzeigen & herunterladen
# ═══════════════════════════════════════════════════════════════

st.divider()
st.subheader("📁 Gespeicherte Dokumente")

doc_by_name = {}
for docs_dir in DOC_SEARCH_DIRS:
    if docs_dir.exists():
        for path in docs_dir.glob("*.pdf"):
            doc_by_name[path.name] = path
doc_paths = sorted(doc_by_name.values(), key=lambda p: p.name.lower())

if doc_paths:
    for doc_path in doc_paths:
        col_name, col_download = st.columns([4, 1])

        with col_name:
            st.write(f"📄 {doc_path.name}")
        with col_download:
            st.download_button(
                "📥 Download",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="application/pdf",
                key=f"download_btn::{doc_path.name}",
            )
else:
    st.caption("Noch keine Dokumente gespeichert.")

# ═══════════════════════════════════════════════════════════════
# Gespeicherte Chunks: anzeigen
# ═══════════════════════════════════════════════════════════════

st.divider()
st.subheader("🧩 Gespeicherte Chunks")

chunks = store.list_chunks()
if chunks:
    st.caption(f"{len(chunks)} Chunks in der Vector-DB.")
    for i, (chunk_text, meta) in enumerate(chunks):
        source = (meta or {}).get("source", "unbekannt")
        with st.expander(f"Chunk {i} · {source} · {len(chunk_text)} Zeichen"):
            st.text(chunk_text)
else:
    st.caption("Noch keine Chunks gespeichert.")

# ═══════════════════════════════════════════════════════════════
# Sidebar: Info
# ═══════════════════════════════════════════════════════════════

st.sidebar.subheader("ℹ️ Über diese App")
st.sidebar.markdown("""
**RAG Agent — LangGraph** demonstriert:
1. **PDF-Upload** & Text-Extraktion
2. **Chunking** mit überlappenden Fenstern
3. **Embedding** via Hash-basiertem Vektor
4. **Dauerhafte Speicherung** in einer Vector-DB (ChromaDB)
5. **Retrieval** mit Kosinus-Ähnlichkeit
6. **Visualisierung** der Ergebnisse

**Erweiterungen (in Produktion):**
- Echte Embedding-Modelle (sentence-transformers)
- LangGraph für Multi-Step Reasoning
- Agentic RAG mit Verify-Retrieve-Refine
- Knowledge Graph Integration
""")

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Vector-DB")
st.sidebar.metric("Dauerhaft gespeicherte Chunks", store.count())
if reindex_stats.get("reindexed") or reindex_stats.get("deleted_only"):
    st.sidebar.caption(
        f"Reindex: {reindex_stats['reindexed']} PDFs neu eingebettet, "
        f"{reindex_stats['deleted_only']} Fehler-Chunks entfernt."
    )

if store.count() > 0 and st.sidebar.button("🗑️ Alle gespeicherten Dokumente löschen"):
    store.clear()
    for docs_dir in DOC_SEARCH_DIRS:
        if docs_dir.exists():
            for doc_path in docs_dir.glob("*.pdf"):
                doc_path.unlink()
    st.sidebar.success("Vector-DB geleert.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("RAG Agent · Streamlit App")

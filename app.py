"""
Streamlit-App: RAG Agent LangGraph
==================================
PDF hochladen → embedden → Fragen stellen, Retrieval visualisieren.
"""

import streamlit as st
import numpy as np
import re
import os
import hashlib
from typing import List, Dict, Tuple

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Agent — LangGraph",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Agent — LangGraph")
st.markdown("PDF hochladen · Embedding · Fragen stellen · Retrieval visualisieren")

# ── Session State ────────────────────────────────────────────
if "documents" not in st.session_state:
    st.session_state.documents = []  # List of {"text": str, "source": str, "chunk_id": int}
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None  # numpy array
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ═══════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ═══════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrahiert Text aus PDF-Bytes (einfache Methode)."""
    text = ""
    try:
        # Versuche PyMuPDF
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except ImportError:
        # Fallback: einfache Text-Extraktion aus PDF-Rohdaten
        content = file_bytes.decode("latin-1", errors="ignore")
        # Suche nach Text zwischen stream/endstream
        text_parts = re.findall(r'BT\s*(.*?)\s*ET', content, re.DOTALL)
        for part in text_parts:
            # Extrahiere Text aus Tj/TJ-Operatoren
            tj_texts = re.findall(r'\((.*?)\)\s*Tj', part)
            text += " ".join(tj_texts) + "\n"
        if not text.strip():
            text = "⚠️ Kein Text extrahierbar. Bitte PyMuPDF installieren: pip install pymupdf"
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Teilt Text in überlappende Chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def simple_embed(text: str, dim: int = 128) -> np.ndarray:
    """Erzeugt ein einfaches Embedding via Hashing + TF-IDF-ähnlicher Gewichtung."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return np.zeros(dim)

    vec = np.zeros(dim)
    for i, word in enumerate(words):
        h = int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16)
        idx = h % dim
        # TF-IDF-ähnlich: häufige Wörter bekommen weniger Gewicht
        vec[idx] += 1.0 / (1.0 + i * 0.01)
    # Normalisieren
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Berechnet Kosinus-Ähnlichkeit."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def retrieve(query: str, chunks: List[str], embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, str]]:
    """Retrieval: Findet die top-k ähnlichsten Chunks."""
    query_emb = simple_embed(query, dim=embeddings.shape[1])
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_emb, emb)
        similarities.append((i, sim, chunks[i]))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


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
        embed_dim = st.selectbox("Embedding-Dimension", [64, 128, 256], index=1)

        if st.button("🔨 PDF verarbeiten & embedden", type="primary"):
            with st.spinner("📄 Extrahiere Text aus PDF..."):
                text = extract_text_from_pdf(file_bytes)
                st.session_state.full_text = text

            with st.spinner("✂️ Erstelle Chunks..."):
                chunks = chunk_text(text, chunk_size, overlap)
                st.session_state.chunks = chunks

            with st.spinner("🧮 Berechne Embeddings..."):
                embeddings = np.array([simple_embed(c, embed_dim) for c in chunks])
                st.session_state.embeddings = embeddings

            st.success(f"✅ {len(chunks)} Chunks erstellt & embedded!")
            st.metric("Chunks", len(chunks))
            st.metric("Embedding-Dim", embed_dim)
            st.metric("Textlänge", f"{len(text):,} Zeichen")

            # Zeige Text-Vorschau
            with st.expander("📝 Text-Vorschau (erste 1000 Zeichen)"):
                st.text(text[:1000])

with col2:
    st.subheader("❓ Frage stellen")

    if st.session_state.embeddings is not None and len(st.session_state.chunks) > 0:
        query = st.text_input("Deine Frage", placeholder="Worum geht es im Dokument?")

        top_k = st.slider("Anzahl Ergebnisse (Top-K)", 1, 10, 5)

        if st.button("🔍 Suchen", type="primary") and query:
            with st.spinner("🔍 Retrieval läuft..."):
                results = retrieve(query, st.session_state.chunks, st.session_state.embeddings, top_k)

            st.divider()
            st.subheader(f"📊 Top-{top_k} Ergebnisse")

            # ── Visualisierung: Ähnlichkeits-Balken ──────────
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 3))
            chunk_labels = [f"Chunk {r[0]}" for r in results]
            sim_values = [r[1] for r in results]
            colors = plt.cm.Blues(np.array(sim_values) / max(sim_values) if max(sim_values) > 0 else 1)
            bars = ax.barh(chunk_labels[::-1], sim_values[::-1], color=colors[::-1])
            ax.set_xlabel("Kosinus-Ähnlichkeit")
            ax.set_title(f"Retrieval-Ergebnisse für: '{query[:50]}...'")
            for bar, val in zip(bars, sim_values[::-1]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center")
            st.pyplot(fig)

            # ── Ergebnisse im Detail ─────────────────────────
            for i, (chunk_id, sim, text) in enumerate(results):
                with st.expander(f"📌 Chunk {chunk_id} — Ähnlichkeit: {sim:.4f}", expanded=(i == 0)):
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
                st.metric("Chunks > 0.1", f"{sum(1 for s in sim_values if s > 0.1)}/{top_k}")

    else:
        st.info("👈 Lade zuerst eine PDF-Datei hoch und klicke auf 'PDF verarbeiten & embedden'.")

# ═══════════════════════════════════════════════════════════════
# Sidebar: Info
# ═══════════════════════════════════════════════════════════════

st.sidebar.subheader("ℹ️ Über diese App")
st.sidebar.markdown("""
**RAG Agent — LangGraph** demonstriert:
1. **PDF-Upload** & Text-Extraktion
2. **Chunking** mit überlappenden Fenstern
3. **Embedding** via Hash-basiertem Vektor
4. **Retrieval** mit Kosinus-Ähnlichkeit
5. **Visualisierung** der Ergebnisse

**Erweiterungen (in Produktion):**
- Echte Embedding-Modelle (sentence-transformers)
- LangGraph für Multi-Step Reasoning
- Agentic RAG mit Verify-Retrieve-Refine
- Knowledge Graph Integration
""")

st.sidebar.metric("Geladene Chunks", len(st.session_state.chunks))
if st.session_state.embeddings is not None:
    st.sidebar.metric("Embedding-Shape", str(st.session_state.embeddings.shape))

st.sidebar.markdown("---")
st.sidebar.caption("RAG Agent · Streamlit App")

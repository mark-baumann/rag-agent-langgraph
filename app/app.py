"""
Streamlit-App: RAG Agent LangGraph
==================================
PDF hochladen → embedden → dauerhaft in Vector-DB speichern → Fragen stellen,
Retrieval visualisieren. Gespeicherte Dokumente überleben App-Neustarts und
Redeploys (Vector-DB liegt auf einem persistenten Volume).
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector_store import DEFAULT_PERSIST_DIR, HashEmbedder, PersistentVectorStore  # noqa: E402

EMBED_DIM = 256

# Original-PDFs werden neben der Vector-DB auf demselben persistenten
# Volume abgelegt, damit sie im Browser angezeigt oder heruntergeladen
# werden können (die Vector-DB enthält nur Text-Chunks, keine Rohdaten).
DOCS_DIR = Path(DEFAULT_PERSIST_DIR).parent / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Agent — LangGraph",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Agent — LangGraph")
st.markdown("PDF hochladen · Embedding · dauerhafte Speicherung · Fragen stellen · Retrieval visualisieren")


@st.cache_resource
def get_store() -> PersistentVectorStore:
    """Öffnet die persistente Vector-DB (ChromaDB, einmal pro Prozess)."""
    return PersistentVectorStore()


@st.cache_resource
def get_embedder() -> HashEmbedder:
    return HashEmbedder(dim=EMBED_DIM)


store = get_store()
embedder = get_embedder()

# ═══════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ═══════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrahiert Text aus PDF-Bytes.

    Versucht der Reihe nach PyMuPDF (fitz), pypdf und — für gescannte PDFs
    ohne Text-Layer — OCR via Tesseract (PyMuPDF `get_textpage_ocr`). Gibt den
    extrahierten Text zurück oder einen leeren String, wenn nichts extrahiert
    werden konnte.
    """
    # 1) PyMuPDF (fitz) — beste Extraktion, inkl. Layout
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        if text.strip():
            return text
    except Exception:
        pass

    # 2) pypdf — reiner Text-Layer, ohne Layout
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if text.strip():
            return text
    except Exception:
        pass

    # 3) OCR (Tesseract) — für gescannte PDFs ohne Text-Layer
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            # full=True erzwingt OCR der kompletten Seite (auch wenn ein
            # (leerer) Text-Layer vorhanden ist). deu+eng deckt deutsche und
            # englische Dokumente ab.
            tp = page.get_textpage_ocr(language="deu+eng", dpi=200, full=True)
            pages_text.append(tp.extractText())
        doc.close()
        text = "\n".join(pages_text)
        if text.strip():
            return text
    except Exception:
        pass

    return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
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

                # Original-PDF ebenfalls dauerhaft ablegen, damit sie später
                # angezeigt oder heruntergeladen werden kann.
                safe_name = Path(uploaded_file.name).name
                (DOCS_DIR / safe_name).write_bytes(file_bytes)

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

doc_paths = sorted(DOCS_DIR.glob("*.pdf"), key=lambda p: p.name.lower())

if doc_paths:
    for doc_path in doc_paths:
        col_name, col_view, col_download = st.columns([4, 1, 1])
        view_key = f"show_preview::{doc_path.name}"

        with col_name:
            st.write(f"📄 {doc_path.name}")
        with col_view:
            if st.button("👁️ Anzeigen", key=f"view_btn::{doc_path.name}"):
                st.session_state[view_key] = not st.session_state.get(view_key, False)
        with col_download:
            st.download_button(
                "📥 Download",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="application/pdf",
                key=f"download_btn::{doc_path.name}",
            )

        if st.session_state.get(view_key, False):
            b64_pdf = base64.b64encode(doc_path.read_bytes()).decode()
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{b64_pdf}" '
                f'width="100%" height="600" style="border:none;"></iframe>',
                unsafe_allow_html=True,
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

if store.count() > 0 and st.sidebar.button("🗑️ Alle gespeicherten Dokumente löschen"):
    store.clear()
    for doc_path in DOCS_DIR.glob("*.pdf"):
        doc_path.unlink()
    st.sidebar.success("Vector-DB geleert.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("RAG Agent · Streamlit App")

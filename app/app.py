"""
Streamlit-App: RAG Agent LangGraph
==================================
PDF hochladen → embedden → dauerhaft in Vector-DB speichern → Fragen stellen,
Retrieval visualisieren. Gespeicherte Dokumente überleben App-Neustarts und
Redeploys (Vector-DB liegt auf einem persistenten Volume).
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from answer_generation import generate_answer  # noqa: E402
from page_explainer import (  # noqa: E402
    LEVELS,
    answer_page_question,
    explain_page,
    extract_page_text,
    page_count,
    render_page_png,
)
from pdf_text import (  # noqa: E402
    OCR_ENGINES,
    chunk_text,
    collect_pdfs,
    extract_text_from_pdf,
    reindex_poisoned_documents,
)
from podcast import LENGTHS as PODCAST_LENGTHS  # noqa: E402
from podcast import generate_podcast_script, synthesize_podcast  # noqa: E402
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


def list_documents() -> list[Path]:
    """Alle gespeicherten Original-PDFs, alphabetisch sortiert."""
    return sorted(collect_pdfs(DOC_SEARCH_DIRS).values(), key=lambda p: p.name.lower())


def save_pdf(name: str, file_bytes: bytes) -> str:
    safe_name = Path(name).name
    for docs_dir in DOC_SEARCH_DIRS:
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / safe_name).write_bytes(file_bytes)
    return safe_name


# ── Lernmodus: Caches ────────────────────────────────────────
# Erklärungen werden auf dem persistenten Volume gespeichert, damit eine
# Seite nicht bei jedem Öffnen erneut (kostenpflichtig) erklärt werden muss.
EXPLANATIONS_DIR = Path(DEFAULT_PERSIST_DIR).parent / "explanations"
PODCASTS_DIR = Path(DEFAULT_PERSIST_DIR).parent / "podcasts"


def _explanations_file(doc_name: str) -> Path:
    return EXPLANATIONS_DIR / f"{doc_name}.json"


def load_explanation(doc_name: str, page_no: int, level: str) -> str | None:
    try:
        data = json.loads(_explanations_file(doc_name).read_text())
    except (OSError, ValueError):
        return None
    return data.get(f"{page_no}:{level}")


def store_explanation(doc_name: str, page_no: int, level: str, text: str) -> None:
    path = _explanations_file(doc_name)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    data[f"{page_no}:{level}"] = text
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


@st.cache_data(show_spinner=False, max_entries=200)
def cached_page_png(path: str, mtime: float, page_index: int) -> bytes:
    return render_page_png(Path(path).read_bytes(), page_index)


@st.cache_data(show_spinner=False, max_entries=200)
def cached_page_text(path: str, mtime: float, page_index: int) -> str:
    return extract_page_text(Path(path).read_bytes(), page_index)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_document_text(path: str, mtime: float) -> str:
    return extract_text_from_pdf(Path(path).read_bytes())


@st.cache_data(show_spinner=False, max_entries=50)
def cached_page_count(path: str, mtime: float) -> int:
    return page_count(Path(path).read_bytes())


def _step_page(key: str, delta: int, total: int) -> None:
    st.session_state[key] = min(max(1, st.session_state.get(key, 1) + delta), total)


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════

tab_learn, tab_rag = st.tabs(["📖 Lernmodus — Seite für Seite erklärt", "🔍 Dokumente durchsuchen (RAG)"])

with tab_learn:
    st.markdown(
        "Lade ein Dokument hoch, das du nicht verstehst. Links siehst du die "
        "Seite, rechts erklärt dir die KI, **was darauf steht** und **was es "
        "bedeutet** — mit Begriffserklärungen und Verständnisfragen."
    )

    learn_upload = st.file_uploader("PDF hochladen", type=["pdf"], key="learn_uploader")
    if learn_upload is not None:
        existing = {p.name for p in list_documents()}
        if Path(learn_upload.name).name not in existing:
            upload_bytes = learn_upload.getvalue()
            with st.spinner("📄 Speichere Dokument & lege Suchindex an..."):
                saved_name = save_pdf(learn_upload.name, upload_bytes)
                full_text = extract_text_from_pdf(upload_bytes)
                if full_text.strip():
                    parts = chunk_text(full_text)
                    store.add(
                        parts,
                        [embedder.embed(part) for part in parts],
                        metadatas=[{"source": saved_name} for _ in parts],
                    )
            st.session_state["learn_doc"] = saved_name

    learn_docs = list_documents()
    if not learn_docs:
        st.info("👆 Lade zuerst ein PDF hoch.")
    else:
        names = [p.name for p in learn_docs]
        if st.session_state.get("learn_doc") not in names:
            st.session_state["learn_doc"] = names[0]
        doc_name = st.selectbox("📄 Dokument", names, key="learn_doc")
        doc_path = learn_docs[names.index(doc_name)]
        mtime = doc_path.stat().st_mtime

        try:
            total_pages = cached_page_count(str(doc_path), mtime)
        except Exception as exc:  # beschädigte PDF
            st.error(f"⚠️ PDF konnte nicht geöffnet werden: {exc}")
            total_pages = 0

        if total_pages:
            page_key = f"learn_page::{doc_name}"
            st.session_state[page_key] = min(st.session_state.get(page_key, 1), total_pages)

            nav_prev, nav_num, nav_next, nav_level = st.columns([1, 2, 1, 4])
            with nav_prev:
                st.button(
                    "◀ Zurück",
                    on_click=_step_page,
                    args=(page_key, -1, total_pages),
                    disabled=st.session_state[page_key] <= 1,
                    width="stretch",
                )
            with nav_num:
                page_no = st.number_input(
                    f"Seite (von {total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    step=1,
                    key=page_key,
                    label_visibility="collapsed",
                )
            with nav_next:
                st.button(
                    "Weiter ▶",
                    on_click=_step_page,
                    args=(page_key, 1, total_pages),
                    disabled=st.session_state[page_key] >= total_pages,
                    width="stretch",
                )
            with nav_level:
                level = st.radio(
                    "Erklär-Niveau",
                    options=list(LEVELS),
                    index=0,
                    format_func={
                        "einfach": "🧒 Ganz einfach",
                        "normal": "🙂 Normal",
                        "fachlich": "🎓 Fachlich",
                    }.get,
                    horizontal=True,
                    key="learn_level",
                    label_visibility="collapsed",
                )

            st.progress(page_no / total_pages, text=f"Seite {page_no} von {total_pages}")

            page_index = page_no - 1
            page_png = cached_page_png(str(doc_path), mtime, page_index)
            page_text = cached_page_text(str(doc_path), mtime, page_index)

            col_page, col_explain = st.columns([1, 1], gap="large")

            with col_page:
                st.image(page_png, width="stretch")
                with st.expander("📝 Text dieser Seite"):
                    st.text(page_text or "(kein Text erkannt)")

            with col_explain:
                st.subheader("🧠 Erklärung")
                has_key = bool(os.environ.get("OPENAI_API_KEY"))
                explanation = load_explanation(doc_name, page_no, level)
                auto = st.toggle("Beim Umblättern automatisch erklären", value=True, key="learn_auto")

                # Auto-Erklärung nur einmal pro Seite versuchen, damit ein
                # Fehler nicht bei jeder Interaktion erneut angefragt wird.
                attempted = st.session_state.setdefault("learn_attempted", set())
                attempt_id = (doc_name, page_no, level)

                regenerate = False
                explain_btn = st.empty()
                if explanation is None:
                    clicked = explain_btn.button("🧠 Diese Seite erklären", type="primary", disabled=not has_key)
                    regenerate = has_key and (clicked or (auto and attempt_id not in attempted))
                elif explain_btn.button("🔄 Neu erklären", disabled=not has_key):
                    regenerate = True

                if regenerate:
                    attempted.add(attempt_id)
                    with st.spinner("🧠 Die KI liest die Seite und erklärt sie..."):
                        try:
                            explanation = explain_page(
                                page_text,
                                page_no,
                                total_pages,
                                doc_name,
                                level=level,
                                page_png=page_png,
                            )
                        except Exception as exc:
                            st.error(f"⚠️ Fehler bei der KI-Anfrage: {exc}")
                            explanation = None
                    if explanation:
                        store_explanation(doc_name, page_no, level, explanation)
                        explain_btn.empty()

                if explanation:
                    st.markdown(explanation)
                elif not has_key:
                    st.warning(
                        "⚠️ Für Erklärungen braucht die App ein Sprachmodell. "
                        "Setze `OPENAI_API_KEY` als Umgebungsvariable im "
                        "Deployment — bis dahin wird nur die Seite angezeigt."
                    )

                # ── Rückfragen zur Seite ─────────────────────
                st.divider()
                st.markdown("**❓ Noch etwas unklar? Frag nach:**")
                chat_key = f"learn_chat::{doc_name}::{page_no}"
                history = st.session_state.setdefault(chat_key, [])
                for role, msg in history:
                    with st.chat_message(role):
                        st.markdown(msg)

                with st.form(key=f"learn_q_form::{doc_name}::{page_no}", clear_on_submit=True):
                    question = st.text_input(
                        "Deine Frage zu dieser Seite",
                        placeholder="z. B. Was bedeutet die Tabelle unten?",
                        label_visibility="collapsed",
                    )
                    asked = st.form_submit_button("Fragen", disabled=not has_key)

                if asked and question.strip():
                    with st.spinner("🧠 Denke nach..."):
                        try:
                            reply = answer_page_question(
                                question,
                                page_text,
                                page_no,
                                doc_name,
                                level=level,
                                page_png=page_png,
                            )
                        except Exception as exc:
                            reply = f"⚠️ Fehler bei der KI-Anfrage: {exc}"
                    history.append(("user", question.strip()))
                    history.append(("assistant", reply or "Keine Antwort erhalten."))
                    st.rerun()

            # ── Podcast ──────────────────────────────────────
            st.divider()
            st.subheader("🎙️ Als Podcast anhören")
            st.caption(
                "Lena und Max unterhalten sich über den Inhalt — Lena fragt das, "
                "was du dich auch fragst, Max erklärt es mit Alltagsbeispielen."
            )
            pod_scope_col, pod_len_col = st.columns(2)
            with pod_scope_col:
                pod_scope = st.radio(
                    "Worüber?",
                    options=["page", "document"],
                    format_func={"page": f"📄 Diese Seite ({page_no})", "document": "📚 Ganzes Dokument"}.get,
                    horizontal=True,
                    key="pod_scope",
                )
            with pod_len_col:
                pod_length = st.radio(
                    "Länge",
                    options=list(PODCAST_LENGTHS),
                    format_func={"kurz": "⚡ Kurz (~3 Min.)", "lang": "🎧 Ausführlich (~8 Min.)"}.get,
                    horizontal=True,
                    key="pod_length",
                )

            scope_id = f"s{page_no}" if pod_scope == "page" else "doc"
            pod_base = PODCASTS_DIR / f"{doc_name}__{scope_id}__{pod_length}__{level}"
            pod_mp3, pod_json = pod_base.with_suffix(".mp3"), pod_base.with_suffix(".json")

            has_podcast = pod_mp3.exists() and pod_json.exists()
            make_podcast = st.button(
                "🔄 Neuen Podcast erstellen" if has_podcast else "🎙️ Podcast erstellen",
                type="secondary" if has_podcast else "primary",
                disabled=not has_key,
            )
            if not has_key:
                st.caption("Braucht `OPENAI_API_KEY` (Skript + Sprachausgabe).")

            if make_podcast:
                if pod_scope == "page":
                    source_text, scope_label = page_text, f"Seite {page_no} von {total_pages}"
                else:
                    source_text = cached_document_text(str(doc_path), mtime)
                    scope_label = f"ganzes Dokument, {total_pages} Seiten"
                try:
                    with st.spinner("✍️ Schreibe das Podcast-Skript..."):
                        script = generate_podcast_script(
                            source_text,
                            doc_name,
                            scope_label,
                            length=pod_length,
                            level_hint=LEVELS[level],
                        )
                    if script is None:
                        st.error("⚠️ Kein Skript erhalten — enthält die Auswahl lesbaren Text?")
                    else:
                        bar = st.progress(0.0, text="🎙️ Nehme die Stimmen auf...")
                        audio = synthesize_podcast(
                            script,
                            on_progress=lambda f: bar.progress(f, text=f"🎙️ Nehme die Stimmen auf... {f:.0%}"),
                        )
                        bar.empty()
                        if audio:
                            PODCASTS_DIR.mkdir(parents=True, exist_ok=True)
                            pod_mp3.write_bytes(audio)
                            pod_json.write_text(json.dumps(script, ensure_ascii=False, indent=1))
                            has_podcast = True
                except Exception as exc:
                    st.error(f"⚠️ Fehler beim Erstellen des Podcasts: {exc}")

            if has_podcast:
                script = json.loads(pod_json.read_text())
                audio = pod_mp3.read_bytes()
                st.markdown(f"#### 🎧 {script['title']}")
                st.audio(audio, format="audio/mpeg")
                st.download_button(
                    "📥 Podcast herunterladen (MP3)",
                    data=audio,
                    file_name=f"{Path(doc_name).stem}_{scope_id}_podcast.mp3",
                    mime="audio/mpeg",
                )
                with st.expander("📜 Transkript"):
                    for seg in script["segments"]:
                        st.markdown(f"**{seg['speaker']}:** {seg['text']}")

with tab_rag:
    # ═══════════════════════════════════════════════════════════════
    # UI
    # ═══════════════════════════════════════════════════════════════

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📄 PDF-Upload")
        uploaded_file = st.file_uploader("PDF-Datei auswählen", type=["pdf"])

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()

            ocr_engine = st.select_slider(
                "🧠 OCR-Engine",
                options=list(OCR_ENGINES),
                value="tesseract",
                help=(
                    "Tesseract: CPU-OCR für Text-Layer und Scans (Standard). "
                    "DeepSeek-OCR: Vision-Modell mit besserer Handschrift-/Layout-"
                    "Erkennung — braucht GPU + Modellgewichte und fällt sonst auf "
                    "Tesseract zurück."
                ),
            )
            chunk_size = st.slider("Chunk-Größe (Wörter)", 100, 1000, 500, 50)
            overlap = st.slider("Überlappung (Wörter)", 0, 300, 100, 25)

            if st.button("💾 PDF verarbeiten & dauerhaft speichern", type="primary"):
                with st.spinner(f"📄 Extrahiere Text aus PDF (OCR: {ocr_engine})..."):
                    text = extract_text_from_pdf(file_bytes, engine=ocr_engine)

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

                    save_pdf(uploaded_file.name, file_bytes)

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

                # ── Generierte Antwort ────────────────────────────
                st.subheader("🧠 Antwort")
                with st.spinner("🧠 Formuliere Antwort..."):
                    answer = generate_answer(query, [r[0] for r in results])

                if answer:
                    st.markdown(answer)
                else:
                    st.warning(
                        "⚠️ Es konnte keine Antwort generiert werden — es wurden "
                        "nur die ähnlichsten Text-Ausschnitte gefunden (unten), "
                        "aber kein Sprachmodell hat sie zu einer Antwort "
                        "zusammengefasst. Setze `OPENAI_API_KEY` als Umgebungs-"
                        "variable im Deployment, damit die App eine echte "
                        "Antwort formulieren kann, statt nur Rohdaten anzuzeigen."
                    )

                st.divider()
                st.subheader(f"📊 Top-{len(results)} Ergebnisse (Rohdaten)")
                st.caption(
                    "Die Ähnlichkeits-Scores kommen aus einem einfachen Hash-"
                    "Embedder (Demo-Zweck) und sind nicht direkt mit Werten "
                    "aus echten Embedding-Modellen vergleichbar — Werte um "
                    "0.3 sind hier normal und heißen nicht automatisch, dass "
                    "kein passender Chunk dabei ist."
                )

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

    doc_paths = list_documents()

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

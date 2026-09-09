from pdf_text import chunk_text, is_poison_chunk, reindex_poisoned_documents
from vector_store import HashEmbedder, PersistentVectorStore

POISON = "⚠️ Kein Text extrahierbar. Bitte PyMuPDF installieren: pip install pymupdf"


def test_is_poison_chunk_detects_install_hint():
    assert is_poison_chunk(POISON)
    assert not is_poison_chunk("Gesamtfinanzierungsbedarf und Ausbildungszuschläge 2026")
    assert not is_poison_chunk("")


def test_chunk_text_overlap():
    text = " ".join(f"w{i}" for i in range(20))
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks[0].startswith("w0 ")
    assert "w9" in chunks[0]
    assert chunks[1].startswith("w8 ")


def test_reindex_replaces_poison_with_extracted_text(tmp_path):
    store = PersistentVectorStore(persist_directory=str(tmp_path / "chroma"))
    embedder = HashEmbedder(dim=32)
    source = "251028_BB_KLM_APD.pdf"
    store.add([POISON], [embedder.embed(POISON)], metadatas=[{"source": source}])

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    pdf_path = docs_dir / source
    pdf_path.write_bytes(b"%PDF-fake")

    stats = reindex_poisoned_documents(
        store,
        embedder,
        [docs_dir],
        extract_fn=lambda _: "Ausbildungszuschläge 2026 Gesamtfinanzierungsbedarf",
    )

    assert stats == {"poisoned": 1, "reindexed": 1, "deleted_only": 0, "failed": 0}
    chunks = [text for text, _ in store.list_chunks()]
    assert chunks == ["Ausbildungszuschläge 2026 Gesamtfinanzierungsbedarf"]
    assert POISON not in chunks


def test_reindex_deletes_poison_when_pdf_missing(tmp_path):
    store = PersistentVectorStore(persist_directory=str(tmp_path / "chroma"))
    embedder = HashEmbedder(dim=32)
    store.add([POISON], [embedder.embed(POISON)], metadatas=[{"source": "gone.pdf"}])

    stats = reindex_poisoned_documents(store, embedder, [tmp_path / "documents"])

    assert stats["poisoned"] == 1
    assert stats["deleted_only"] == 1
    assert store.count() == 0


def test_reindex_noop_when_clean(tmp_path):
    store = PersistentVectorStore(persist_directory=str(tmp_path / "chroma"))
    embedder = HashEmbedder(dim=32)
    store.add(["echter dokumenttext"], [embedder.embed("echter dokumenttext")])

    stats = reindex_poisoned_documents(store, embedder, [tmp_path / "documents"])

    assert stats["poisoned"] == 0
    assert store.count() == 1

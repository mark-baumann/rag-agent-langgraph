# ═══════════════════════════════════════════════════════════════
# Dockerfile — Multi-Stage-Build für RAG Agent (rag-agent-langgraph)
# ═══════════════════════════════════════════════════════════════
# Abweichung vom Standard-Template (Dockerfile.template): git/perl
# werden nur im Builder installiert (für die eine git+https-Pip-
# Abhängigkeit), das finale Image enthält sie nicht mehr. Grund:
# das Perl-Paket bringt tausende kleiner Dateien mit, was auf dem
# Pi wiederholt zu "failed to Lchown ... no such file or directory"
# beim Layer-Extract in containerd/overlayfs führte.

# ── Builder ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Final Image ──────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Tesseract OCR + deutsche/englische Sprachdaten — für gescannte PDFs ohne
# Text-Layer (PyMuPDF `get_textpage_ocr` nutzt Tesseract als Engine).
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# App-Code
COPY . .

# Persistentes Volume für die Vector-DB (siehe infrastruktur-deployment: volumes-Mount)
ENV VECTOR_DB_PATH=/app/data/chroma
RUN mkdir -p /app/data
VOLUME ["/app/data"]

# Port (pro App anpassen: 8501-8519)
# ARG allein reicht nicht: CMD/HEALTHCHECK laufen zur Container-Laufzeit,
# nicht beim Build, und lesen $PORT vom Shell-Environment der Shell-Form —
# ARG-Werte sind zu dem Zeitpunkt längst weg. Als ENV re-exportieren, damit
# der Wert im laufenden Container tatsächlich gesetzt ist.
ARG PORT=8512
ENV PORT=$PORT
EXPOSE $PORT

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:${PORT}/_stcore/health')"

# Streamlit
CMD streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true

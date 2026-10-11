# Baar — Armenian Contexto. Runtime image (numpy-only, no torch/gensim).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + web UI + precomputed model artifacts.
# The large data/raw corpora and venvs are excluded via .dockerignore;
# the numpy vectors (data/*.npy, data/*.txt) ARE copied in and baked into
# the image so the container is self-contained at runtime.
COPY src/ ./src/
COPY web/ ./web/
COPY data/ ./data/

EXPOSE 8080

# Single worker: the 315 MB vector table is loaded per-process, so extra
# workers would multiply memory. One worker is plenty for this traffic.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]

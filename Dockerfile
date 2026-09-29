FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cortex ./cortex
COPY frontend ./frontend

ENV GRAPH_SNAPSHOT_PATH=/app/data/graph_memory.json \
    CHROMA_PERSIST_DIR=/app/data/chroma
VOLUME ["/app/data"]

EXPOSE 8000
CMD ["uvicorn", "cortex.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Cairn ──────────────────────────────────────────
#
# Build:  docker build -t cairn .
# Run:    docker compose up
#
# GPU passthrough: add --gpus all to docker run or set in compose.
# Requires nvidia-container-toolkit installed on the host.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY server/ server/
COPY pipeline/ pipeline/
COPY core/ core/
COPY throttle/ throttle/
COPY cli/ cli/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENV OLLAMA_BASE_URL=http://ollama:11434

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]
